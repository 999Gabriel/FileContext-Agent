import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Set, Callable
from config import config


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, on_file_changed: Callable[[str, str], None]):
        self.on_file_changed = on_file_changed
        self.allowed_extensions = set(config.allowed_extensions)

    def _is_allowed_file(self, file_path: str) -> bool:
        """Check if file extension is allowed."""
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.allowed_extensions

    def on_modified(self, event):
        if not event.is_directory and self._is_allowed_file(event.src_path):
            self.on_file_changed("modified", event.src_path)

    def on_created(self, event):
        if not event.is_directory and self._is_allowed_file(event.src_path):
            self.on_file_changed("created", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and self._is_allowed_file(event.src_path):
            self.on_file_changed("deleted", event.src_path)


class FileWatcher:
    def __init__(self, embedder, retriever):
        self.embedder = embedder
        self.retriever = retriever
        self.observer = Observer()
        self.is_running = False

    def _handle_file_change(self, event_type: str, file_path: str):
        """Handle file system changes."""
        print(f"File {event_type}: {file_path}")

        if event_type == "deleted":
            self.retriever.remove_file_documents(file_path)
        else:  # modified or created
            # Remove old documents for this file
            self.retriever.remove_file_documents(file_path)

            # Process and add new documents
            documents = self.embedder.process_file(file_path)
            if documents:
                self.retriever.add_documents(documents)

        # Save changes
        self.retriever.save_index()

    def start_watching(self, folder_path: str = None):
        """Start watching the specified folder."""
        if folder_path is None:
            folder_path = config.project_folder

        event_handler = FileChangeHandler(self._handle_file_change)
        self.observer.schedule(event_handler, folder_path, recursive=True)

        self.observer.start()
        self.is_running = True
        print(f"Started watching folder: {folder_path}")

    def stop_watching(self):
        """Stop watching files."""
        if self.is_running:
            self.observer.stop()
            self.observer.join()
            self.is_running = False
            print("Stopped watching files")

    def scan_and_index_all(self, folder_path: str = None):
        """Initial scan of all files in the folder."""
        if folder_path is None:
            folder_path = config.project_folder

        print(f"Scanning folder: {folder_path}")

        for root, dirs, files in os.walk(folder_path):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)

                if ext.lower() in config.allowed_extensions:
                    documents = self.embedder.process_file(file_path)
                    if documents:
                        self.retriever.add_documents(documents)
                        print(f"Indexed: {file_path}")

        self.retriever.save_index()
        print("Initial indexing complete")