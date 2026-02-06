import subprocess
import os


class RepoCloner:
    """A utility class to clone a GitHub repository."""

    def __init__(self, repo_url: str, target_dir: str = "peerread"):
        self.repo_url = repo_url
        self.target_dir = target_dir

    def clone(self):
        """Clone the repository into the target directory. Skips if it already exists."""
        if os.path.exists(self.target_dir):
            print(f"[INFO] Directory '{self.target_dir}' already exists, skipping clone.")
            return

        print(f"[INFO] Cloning '{self.repo_url}' into '{self.target_dir}'...")
        try:
            subprocess.run(
                ["git", "clone", self.repo_url, self.target_dir],
                check=True,
            )
            print(f"[OK] Repository successfully cloned into: {self.target_dir}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Clone failed: {e}")
        except FileNotFoundError:
            print("[ERROR] 'git' not found. Please make sure git is installed.")
