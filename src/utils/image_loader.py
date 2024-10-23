import sys
import os
from pathlib import Path
from loguru import logger

class ImageLoader:
    """
    Image Loader capable of finding and getting all images paths in a given folder recursively
    in order to load them into a model
    """
    def __init__(self, folder, verbose=False):

        # Attr initialization
        self.folder = Path(folder)

        # Setup logging
        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        # Checking folder
        try:
            logger.info("Checking provided path.")
            if not self.folder.is_dir():
                raise NotADirectoryError(f"Provided path {self.folder.absolute()} is not a directory")
            logger.info(f"Provided path {self.folder.absolute()} is a valid directory.")
        except NotADirectoryError as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
        except FileNotFoundError:
            logger.error(f"Couldnt find provided path: {self.folder}")
            sys.exit(1)
        except PermissionError:
            logger.error(f"No permission to access provided path: {self.folder}")
            sys.exit(1)



    def find_images(self):
        """Find all image files in path recursively. Return their paths"""
        return (
            list(self.folder.rglob("*.jpg"))
            + list(self.folder.rglob("*.JPG"))
            + list(self.folder.rglob("*.jpeg"))
            + list(self.folder.rglob("*.png"))
            + list(self.folder.rglob("*.bmp"))
        )
