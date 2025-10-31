import sys
import subprocess
import os
import logging

logger = logging.getLogger(__name__)


def try_copy(command: list[str], content: str, encoding: str = "utf-8") -> bool:
    """
    Attempt to copy content to the clipboard using the specified command.
    Returns True if the command executed successfully, or False if the command was not found.
    """
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        process.communicate(input=content.encode(encoding))
        logger.debug(f"Copied content using '{command[0]}'.")
        return True
    except FileNotFoundError:
        logger.debug(f"Command '{command[0]}' not found.")
        return False


def copy_content(content: str) -> None:
    """Copy the given content to the clipboard."""
    if sys.platform == "win32":
        # On Windows, use the clip command with UTF-16LE encoding
        process = subprocess.Popen("clip", stdin=subprocess.PIPE, shell=True)  # pyright: ignore[reportUnreachable]
        process.communicate(input=content.encode("utf-16le"))
        logger.debug("Copied content using 'clip' on Windows.")

    elif sys.platform == "darwin":
        # On macOS, use the pbcopy command with UTF-8 encoding
        process = subprocess.Popen("pbcopy", stdin=subprocess.PIPE)  # pyright: ignore[reportUnreachable]
        process.communicate(input=content.encode("utf-8"))
        logger.debug("Copied content using 'pbcopy' on macOS.")

    elif sys.platform.startswith("linux"):
        # Determine which clipboard utilities to use based on environment variables.
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        wayland = (session_type == "wayland") or bool(os.environ.get("WAYLAND_DISPLAY"))
        x11 = (session_type == "x11") or bool(os.environ.get("DISPLAY"))

        if wayland:
            if try_copy(["wl-copy"], content):
                return
            logger.warning(
                "'wl-copy' not found on Wayland; falling back to X11 clipboard utilities."
            )

        if x11:
            if try_copy(["xclip", "-selection", "clipboard"], content):
                return
            if try_copy(["xsel", "--clipboard", "--input"], content):
                return
            logger.error(
                "Clipboard functionality requires 'xclip' or 'xsel' to be installed on Linux."
            )
            print(
                "Clipboard functionality requires 'xclip' or 'xsel' to be installed on Linux."
            )
        else:
            logger.error("No suitable clipboard environment detected on Linux.")
            print("Clipboard functionality is not supported on this Linux environment.")

    else:
        logger.error(f"Clipboard functionality is not supported on {sys.platform}.")
        print(f"Clipboard functionality is not supported on {sys.platform}.")
