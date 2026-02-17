import os
import shutil

class StorageManager:
    def __init__(self, base_dir='storage'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.audio_dir = os.path.join(self.base_dir, 'audio')
        self.model_dir = os.path.join(self.base_dir, 'models')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        for d in [self.audio_dir, self.model_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

    def save_audio(self, file_path, dest_name):
        dest = os.path.join(self.audio_dir, dest_name)
        shutil.copy(file_path, dest)
        return dest

    def save_model(self, model_path, dest_name):
        dest = os.path.join(self.model_dir, dest_name)
        shutil.copy(model_path, dest)
        return dest

    def save_log(self, log_content, log_name):
        dest = os.path.join(self.log_dir, log_name)
        with open(dest, 'w', encoding='utf-8') as f:
            f.write(log_content)
        return dest
