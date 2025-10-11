"""Model storage service for managing trained model artifacts."""
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class ModelStorageService:
    """Handles storage and retrieval of trained model artifacts."""
    
    def __init__(self, base_dir: str = "/app/models"):
        """
        Initialize model storage service.
        
        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.active_dir = self.base_dir / "active"
        self.archive_dir = self.base_dir / "archive"
        self.temp_dir = self.base_dir / "temp"
        
        for dir_path in [self.active_dir, self.archive_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_id: str, archived: bool = False) -> Path:
        """
        Get the path for a model file.
        
        Args:
            model_id: Unique model identifier
            archived: Whether to look in archive directory
            
        Returns:
            Path to model file
        """
        base_dir = self.archive_dir if archived else self.active_dir
        return base_dir / f"{model_id}.joblib"
    
    def save_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> str:
        """
        Save model metadata as JSON file.
        
        Args:
            model_id: Unique model identifier
            metadata: Model metadata dictionary
            
        Returns:
            Path to metadata file
        """
        metadata_path = self.active_dir / f"{model_id}_metadata.json"
        
        # Add storage timestamp
        metadata['storage_timestamp'] = datetime.utcnow().isoformat()
        metadata['model_id'] = model_id
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(metadata_path)
    
    def load_model_metadata(self, model_id: str, archived: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load model metadata from JSON file.
        
        Args:
            model_id: Unique model identifier
            archived: Whether to look in archive directory
            
        Returns:
            Model metadata dictionary or None if not found
        """
        base_dir = self.archive_dir if archived else self.active_dir
        metadata_path = base_dir / f"{model_id}_metadata.json"
        
        if not metadata_path.exists():
            return None
        
        import json
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load metadata for {model_id}: {e}")
            return None
    
    def model_exists(self, model_id: str, archived: bool = False) -> bool:
        """
        Check if a model exists in storage.
        
        Args:
            model_id: Unique model identifier
            archived: Whether to check archive directory
            
        Returns:
            True if model exists, False otherwise
        """
        model_path = self.get_model_path(model_id, archived)
        return model_path.exists()
    
    def get_model_size(self, model_id: str, archived: bool = False) -> Optional[int]:
        """
        Get the size of a model file in bytes.
        
        Args:
            model_id: Unique model identifier
            archived: Whether to check archive directory
            
        Returns:
            File size in bytes or None if not found
        """
        model_path = self.get_model_path(model_id, archived)
        
        if not model_path.exists():
            return None
        
        return model_path.stat().st_size
    
    def list_models(self, archived: bool = False) -> List[Dict[str, Any]]:
        """
        List all models in storage.
        
        Args:
            archived: Whether to list archived models
            
        Returns:
            List of model information dictionaries
        """
        base_dir = self.archive_dir if archived else self.active_dir
        models = []
        
        # Find all .joblib files
        for model_file in base_dir.glob("*.joblib"):
            model_id = model_file.stem
            
            # Skip metadata files
            if model_id.endswith("_metadata"):
                continue
            
            model_info = {
                'model_id': model_id,
                'file_path': str(model_file),
                'file_size': model_file.stat().st_size,
                'created_at': datetime.fromtimestamp(model_file.stat().st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                'archived': archived
            }
            
            # Load metadata if available
            metadata = self.load_model_metadata(model_id, archived)
            if metadata:
                model_info['metadata'] = metadata
            
            models.append(model_info)
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        return models
    
    def archive_model(self, model_id: str) -> bool:
        """
        Move a model from active to archive storage.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            True if archived successfully, False otherwise
        """
        active_model_path = self.get_model_path(model_id, archived=False)
        archive_model_path = self.get_model_path(model_id, archived=True)
        
        if not active_model_path.exists():
            return False
        
        try:
            # Move model file
            shutil.move(str(active_model_path), str(archive_model_path))
            
            # Move metadata file if it exists
            active_metadata_path = self.active_dir / f"{model_id}_metadata.json"
            archive_metadata_path = self.archive_dir / f"{model_id}_metadata.json"
            
            if active_metadata_path.exists():
                shutil.move(str(active_metadata_path), str(archive_metadata_path))
            
            return True
            
        except Exception as e:
            print(f"Failed to archive model {model_id}: {e}")
            return False
    
    def restore_model(self, model_id: str) -> bool:
        """
        Move a model from archive to active storage.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            True if restored successfully, False otherwise
        """
        archive_model_path = self.get_model_path(model_id, archived=True)
        active_model_path = self.get_model_path(model_id, archived=False)
        
        if not archive_model_path.exists():
            return False
        
        try:
            # Move model file
            shutil.move(str(archive_model_path), str(active_model_path))
            
            # Move metadata file if it exists
            archive_metadata_path = self.archive_dir / f"{model_id}_metadata.json"
            active_metadata_path = self.active_dir / f"{model_id}_metadata.json"
            
            if archive_metadata_path.exists():
                shutil.move(str(archive_metadata_path), str(active_metadata_path))
            
            return True
            
        except Exception as e:
            print(f"Failed to restore model {model_id}: {e}")
            return False
    
    def delete_model(self, model_id: str, archived: bool = False, permanent: bool = False) -> bool:
        """
        Delete a model from storage.
        
        Args:
            model_id: Unique model identifier
            archived: Whether to delete from archive directory
            permanent: If True, delete permanently. If False, move to archive first.
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not permanent and not archived:
            # Move to archive first
            return self.archive_model(model_id)
        
        # Permanent deletion
        base_dir = self.archive_dir if archived else self.active_dir
        model_path = base_dir / f"{model_id}.joblib"
        metadata_path = base_dir / f"{model_id}_metadata.json"
        
        try:
            # Delete model file
            if model_path.exists():
                model_path.unlink()
            
            # Delete metadata file
            if metadata_path.exists():
                metadata_path.unlink()
            
            return True
            
        except Exception as e:
            print(f"Failed to delete model {model_id}: {e}")
            return False
    
    def cleanup_old_models(self, days_old: int = 30, archived_days: int = 90) -> Dict[str, int]:
        """
        Clean up old models based on age.
        
        Args:
            days_old: Archive active models older than this many days
            archived_days: Delete archived models older than this many days
            
        Returns:
            Dictionary with cleanup statistics
        """
        now = datetime.utcnow()
        archive_cutoff = now - timedelta(days=days_old)
        delete_cutoff = now - timedelta(days=archived_days)
        
        archived_count = 0
        deleted_count = 0
        
        # Archive old active models
        for model_info in self.list_models(archived=False):
            created_at = datetime.fromisoformat(model_info['created_at'])
            if created_at < archive_cutoff:
                if self.archive_model(model_info['model_id']):
                    archived_count += 1
        
        # Delete old archived models
        for model_info in self.list_models(archived=True):
            created_at = datetime.fromisoformat(model_info['created_at'])
            if created_at < delete_cutoff:
                if self.delete_model(model_info['model_id'], archived=True, permanent=True):
                    deleted_count += 1
        
        return {
            'archived_count': archived_count,
            'deleted_count': deleted_count
        }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        active_models = self.list_models(archived=False)
        archived_models = self.list_models(archived=True)
        
        active_size = sum(model['file_size'] for model in active_models)
        archived_size = sum(model['file_size'] for model in archived_models)
        
        return {
            'active_models_count': len(active_models),
            'archived_models_count': len(archived_models),
            'total_models_count': len(active_models) + len(archived_models),
            'active_storage_bytes': active_size,
            'archived_storage_bytes': archived_size,
            'total_storage_bytes': active_size + archived_size,
            'active_storage_mb': round(active_size / (1024 * 1024), 2),
            'archived_storage_mb': round(archived_size / (1024 * 1024), 2),
            'total_storage_mb': round((active_size + archived_size) / (1024 * 1024), 2)
        }