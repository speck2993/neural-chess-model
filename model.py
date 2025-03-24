import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np

class SE(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, in_channels, reduction=8):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        fc = self.fc(avg_pool).view(batch_size, channels, 1, 1)
        return x * fc.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual block with SE attention and configurable dropout"""
    def __init__(self, in_channels, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # SE attention for efficient channel weighting
        self.attention = SE(in_channels)
            
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        residual = x
        
        # First conv + bn + relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv + bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention
        out = self.attention(out)
        
        # Add residual connection
        out += residual
        
        # Final activation
        out = F.relu(out)
        
        # Apply dropout if specified
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out
    
class PolicyHead(nn.Module):
    """Policy head predicting move probabilities"""
    def __init__(self, in_channels):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, 73, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        # Note: We don't apply softmax here as it's part of the loss function
        return out
    
class ValueHead(nn.Module):
    """Value head predicting game outcome probabilities"""
    def __init__(self, in_channels):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2*8*8, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # Note: We don't apply softmax here as it's part of the loss function
        return out
    
class ChessResNet(nn.Module):
    """Simplified ResNet for chess position analysis with auxiliary heads"""
    def __init__(self, filters=256, blocks=12, initial_lr=0.01):
        super(ChessResNet, self).__init__()
        
        # Fixed hyperparameters
        self.policy_weight = 1.0
        self.value_weight = 1.6
        self.auxiliary_weight = 0.4
        self.gradient_clip = 1.0
        
        # Gradient accumulation settings
        self.grad_accumulation_steps = 4  # Accumulate over 4 steps
        self.accumulated_steps = 0
        
        # Input processing
        self.conv = nn.Conv2d(17, filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(filters)
        
        # Progressive dropout rates
        dropout_rates = [0.3, 0.2, 0.1] + [0.05] * 9  # First 3 blocks have higher dropout, rest have minimal
        
        # Create blocks in groups to allow for auxiliary heads
        self.early_blocks = nn.ModuleList()
        for i in range(4):  # First 4 blocks
            self.early_blocks.append(ResidualBlock(filters, dropout_rates[i]))
        
        # Auxiliary heads after 4 blocks
        self.auxiliary_policy_head = PolicyHead(filters)
        self.auxiliary_value_head = ValueHead(filters)
        
        # Remaining blocks
        self.mid_blocks = nn.ModuleList()
        for i in range(4, 8):  # Next 4 blocks
            self.mid_blocks.append(ResidualBlock(filters, dropout_rates[i]))
            
        self.late_blocks = nn.ModuleList()
        for i in range(8, blocks):  # Final blocks
            self.late_blocks.append(ResidualBlock(filters, dropout_rates[i]))
        
        # Final heads
        self.policy_head = PolicyHead(filters)
        self.value_head = ValueHead(filters)
        
        # Initialize optimizer with higher learning rate
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=initial_lr,  # Higher initial learning rate
            weight_decay=0.0005
        )
        
        # Initialize AMP scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
    
    def forward(self, x):
        """Forward pass through the network with auxiliary outputs"""
        
        # Initial convolution
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        
        # Early blocks
        for block in self.early_blocks:
            out = block(out)
            
        # Auxiliary outputs after 4 blocks
        aux_policy = self.auxiliary_policy_head(out)
        aux_value = self.auxiliary_value_head(out)
        
        # Mid blocks
        for block in self.mid_blocks:
            out = block(out)
        
        # Late blocks
        for block in self.late_blocks:
            out = block(out)
        
        # Final outputs
        policy = self.policy_head(out)
        value = self.value_head(out)
        
        return policy, value, aux_policy, aux_value
    
    def policy_loss(self, target, output):
        """Log-softmax based policy loss function"""
        # First reshape to match the expected shape
        batch_size = output.size(0)
        output = output.view(batch_size, 73, 8, 8)
        target = target.view(batch_size, 73, 8, 8)
        
        # Flatten spatial dimensions for softmax
        output = output.reshape(batch_size, 73, -1)
        target = target.reshape(batch_size, 73, -1)
        
        # Further reshape for softmax over all possible moves
        output = output.transpose(1, 2).reshape(batch_size, -1)
        target = target.transpose(1, 2).reshape(batch_size, -1)
        
        # Normalize target to sum to 1
        target_sum = target.sum(dim=1, keepdim=True)
        target = target / torch.clamp(target_sum, min=1e-5)
        
        # Apply log_softmax and compute negative log likelihood
        log_prob = F.log_softmax(output, dim=1)
        loss = -(target * log_prob).sum(dim=1).mean()
        
        return loss
    
    def value_loss(self, target, output):
        """Log-softmax based value loss function"""
        log_prob = F.log_softmax(output, dim=1)
        loss = -(target * log_prob).sum(dim=1).mean()
        return loss
    
    def compute_loss(self, policy_out, value_out, aux_policy, aux_value, policy_target, value_target):
        """Compute combined loss with fixed weights"""
        policy_loss = self.policy_loss(policy_target, policy_out)
        value_loss = self.value_loss(value_target, value_out)
        
        aux_policy_loss = self.policy_loss(policy_target, aux_policy)
        aux_value_loss = self.value_loss(value_target, aux_value)
        
        # Use fixed weights with value getting higher weight (1.6) than policy (1.0)
        total_loss = (self.policy_weight * policy_loss + 
                      self.value_weight * value_loss + 
                      self.auxiliary_weight * (self.policy_weight * aux_policy_loss + 
                                             self.value_weight * aux_value_loss))
        
        return total_loss, policy_loss, value_loss
        
    def save(self, path):
        """Save model weights with explicit type information"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'mixed_precision_enabled': True  # Flag to track mixed precision usage
        }, path)
            
    def load(self, path, device='cuda'):
        """Load model weights with type consistency handling"""
        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'
                
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(device)
        
        # Ensure all parameters have consistent dtype
        for param in self.parameters():
            # Force all parameters to float32 before training
            param.data = param.data.float()
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            # Fix optimizer states device placement
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param in self.optimizer.state:
                        for state_name, state_value in self.optimizer.state[param].items():
                            if isinstance(state_value, torch.Tensor):
                                self.optimizer.state[param][state_name] = state_value.to(device).float()
        
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.accumulated_steps = 0
        
    def training_step(self, batch, device='cuda', use_mixed_precision=True):
        # Unpack batch
        bitboards, policies, values = batch
        batch_size = bitboards.size(0)
        
        # Move data to device
        bitboards = bitboards.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # First step in accumulation - zero gradients
        if self.accumulated_steps == 0:
            self.optimizer.zero_grad()
        
        # Mixed precision forward pass
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                # Ensure compatible types between model and input
                policy_out, value_out, aux_policy, aux_value = self(bitboards)
                loss, policy_loss, value_loss = self.compute_loss(
                    policy_out, value_out, aux_policy, aux_value, policies, values
                )
                
                # Calculate individual auxiliary losses for tracking
                aux_policy_loss = self.policy_loss(policies, aux_policy)
                aux_value_loss = self.value_loss(values, aux_value)
                
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accumulation_steps
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
        else:
            # Standard forward pass
            policy_out, value_out, aux_policy, aux_value = self(bitboards)
            loss, policy_loss, value_loss = self.compute_loss(
                policy_out, value_out, aux_policy, aux_value, policies, values
            )
            
            # Calculate individual auxiliary losses for tracking
            aux_policy_loss = self.policy_loss(policies, aux_policy)
            aux_value_loss = self.value_loss(values, aux_value)
            
            # Scale loss for gradient accumulation
            loss = loss / self.grad_accumulation_steps
            
            # Standard backward pass
            loss.backward()
        
        # Increment accumulation step counter
        self.accumulated_steps += 1
        
        # Step optimizer when accumulation is complete
        if self.accumulated_steps >= self.grad_accumulation_steps:
            if use_mixed_precision:
                # Unscale before clip to avoid any numerical issues
                self.scaler.unscale_(self.optimizer)
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
                
                # Step optimizer with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
                
                # Standard optimizer step
                self.optimizer.step()
            
            # Reset accumulation counter
            self.accumulated_steps = 0
        
        # Return metrics for tracking (returning original loss, not the scaled version)
        return {
            'loss': loss.item() * self.grad_accumulation_steps,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'aux_policy_loss': aux_policy_loss.item(),
            'aux_value_loss': aux_value_loss.item()
        }

    def evaluate(self, dataloader, device='cuda', use_mixed_precision=True):
        """Evaluate model on the given dataloader with mixed precision support"""
        self.eval()
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        aux_policy_loss = 0
        aux_value_loss = 0
        correct_moves = 0
        correct_outcomes = 0
        total_samples = 0
        
        device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.to(device)
        
        with torch.no_grad():
            # Use the next_batch method instead of iterating directly
            while True:
                # Get next batch
                batch = dataloader.next_batch()
                if batch is None:
                    break  # End of dataset
                    
                bitboards, policies, values = batch
                
                # Move to device
                bitboards = bitboards.to(device)
                policies = policies.to(device)
                values = values.to(device)
                
                # Forward pass with mixed precision if enabled
                if use_mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        policy_preds, value_preds, aux_policy, aux_value = self(bitboards)
                        loss, curr_policy_loss, curr_value_loss = self.compute_loss(
                            policy_preds, value_preds, aux_policy, aux_value, policies, values
                        )
                        
                        # Calculate individual auxiliary losses for tracking
                        curr_aux_policy_loss = self.policy_loss(policies, aux_policy)
                        curr_aux_value_loss = self.value_loss(values, aux_value)
                else:
                    policy_preds, value_preds, aux_policy, aux_value = self(bitboards)
                    loss, curr_policy_loss, curr_value_loss = self.compute_loss(
                        policy_preds, value_preds, aux_policy, aux_value, policies, values
                    )
                    
                    # Calculate individual auxiliary losses for tracking
                    curr_aux_policy_loss = self.policy_loss(policies, aux_policy)
                    curr_aux_value_loss = self.value_loss(values, aux_value)
                
                # Update metrics
                batch_size = bitboards.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size
                policy_loss += curr_policy_loss.item() * batch_size
                value_loss += curr_value_loss.item() * batch_size
                aux_policy_loss += curr_aux_policy_loss.item() * batch_size
                aux_value_loss += curr_aux_value_loss.item() * batch_size
                
                # Calculate accuracy
                # For policy, find highest probability move
                policy_preds_flat = policy_preds.view(batch_size, 73, 8, 8).reshape(batch_size, -1)
                policies_flat = policies.view(batch_size, 73, 8, 8).reshape(batch_size, -1)
                pred_moves = policy_preds_flat.argmax(dim=1)
                target_moves = policies_flat.argmax(dim=1)
                correct_moves += (pred_moves == target_moves).sum().item()
                
                # For value, find highest probability outcome
                pred_outcomes = F.softmax(value_preds, dim=1).argmax(dim=1)
                target_outcomes = values.argmax(dim=1)
                correct_outcomes += (pred_outcomes == target_outcomes).sum().item()
                
                # Clear memory
                del bitboards, policies, values, policy_preds, value_preds
                del pred_moves, target_moves, pred_outcomes, target_outcomes
                
                # Force garbage collection periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_policy_loss = policy_loss / total_samples if total_samples > 0 else 0
        avg_value_loss = value_loss / total_samples if total_samples > 0 else 0
        avg_aux_policy_loss = aux_policy_loss / total_samples if total_samples > 0 else 0
        avg_aux_value_loss = aux_value_loss / total_samples if total_samples > 0 else 0
        move_accuracy = correct_moves / total_samples if total_samples > 0 else 0
        outcome_accuracy = correct_outcomes / total_samples if total_samples > 0 else 0
        
        return {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'aux_policy_loss': avg_aux_policy_loss,
            'aux_value_loss': avg_aux_value_loss,
            'move_accuracy': move_accuracy,
            'outcome_accuracy': outcome_accuracy,
            'samples': total_samples
        }

    def predict_position(self, bitboard, device='cuda', use_mixed_precision=True):
        """Predict policy and value for a single position"""
        self.eval()
        device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.to(device)
        
        # Convert to tensor and add batch dimension
        bitboard_tensor = torch.tensor(bitboard, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Forward pass with mixed precision if enabled
            if use_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    policy, value, _, _ = self(bitboard_tensor)
            else:
                policy, value, _, _ = self(bitboard_tensor)
            
            # Apply softmax to get probabilities
            policy = F.softmax(policy.view(1, 73, 8, 8).reshape(1, -1), dim=1)
            value = F.softmax(value, dim=1)
            
            # Convert to numpy
            policy_np = policy.cpu().numpy()[0]  # Remove batch dimension
            value_np = value.cpu().numpy()[0]    # Remove batch dimension
            
            # Clean up memory
            del bitboard_tensor, policy, value
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return policy_np, value_np