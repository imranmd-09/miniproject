def generate_summary(args):
    """Generate a video summary using a trained model."""
    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    
    # Create model
    model = TransformerSummarizer(
        feature_dim=2048,  # ResNet50 feature dimension
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Setup feature extractor
    feature_extractor = setup_feature_extractor()
    
    # Create summarizer
    summarizer = VideoSummarizer(
        model=model,
        feature_extractor=feature_extractor
    )
    
    # Generate summary
    frames = summarizer.generate_summary(
        video_path=args.video_path,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        diversity_threshold=args.diversity_threshold
    )
    
    # Save summary
    if args.output_video:
        summarizer.save_summary(frames, args.output_video)
    
    if args.output_dir:
        summarizer.save_frames(frames, args.output_dir)

Train.py

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int,
                learning_rate: float,
                device: torch.device,
                patience: int = 5) -> Dict[str, List[float]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, scores in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features = features.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            pred_scores = model(features)
            loss = criterion(pred_scores, scores)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, scores in val_loader:
                features = features.to(device)
                scores = scores.to(device)
                
                pred_scores = model(features)
                loss = criterion(pred_scores, scores)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_model.pt')
        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter} epochs')
            
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return history
