"""Checkpoint sync utilities for S3/SD storage."""
import os
import subprocess
import threading
import logging

SD_SCRIPT = os.path.expanduser("~/setup/ssd-symlinks.sh")


def _sd(*args, cwd=None):
    """Call sd script directly without sourcing .bashrc."""
    return subprocess.run(
        ['bash', SD_SCRIPT, *args],
        capture_output=True, text=True,
        cwd=cwd or os.getcwd()
    )


def sync_checkpoint_async(checkpoint_dir: str, logger: logging.Logger) -> None:
    """Non-blocking SD sync of checkpoint directory to S3."""
    def sync_with_logging(path):
        result = _sd('sync', path)
        if result.returncode == 0:
            logger.info(f"Successfully synced {path} to S3")
        else:
            logger.warning(f"Failed to sync {path}: {result.stderr}")

    sync_thread = threading.Thread(target=sync_with_logging, args=(checkpoint_dir,))
    sync_thread.daemon = True
    sync_thread.start()
    logger.info(f"Started async SD sync for {checkpoint_dir}")


def sync_checkpoint_blocking(checkpoint_dir: str, logger: logging.Logger) -> bool:
    """Blocking SD sync of checkpoint directory to S3. Returns success status."""
    logger.info(f"Final sync: syncing {checkpoint_dir} to S3...")
    result = _sd('sync', checkpoint_dir)
    if result.returncode == 0:
        logger.info(f"Successfully synced {checkpoint_dir} to S3")
        return True
    else:
        logger.warning(f"Failed to sync {checkpoint_dir}: {result.stderr}")
        return False


def sync_evals_async(eval_dir: str, logger: logging.Logger) -> None:
    """Non-blocking sync of evals to S3. Copies to sevals/ then syncs."""
    def sync_with_logging():
        os.makedirs("sevals", exist_ok=True)
        cp_result = subprocess.run(
            ['bash', '-c', f'cp -r {eval_dir}/* sevals/.'],
            capture_output=True, text=True
        )
        if cp_result.returncode != 0:
            logger.warning(f"Failed to copy evals: {cp_result.stderr}")
            return
        result = _sd('sync', 'sevals')
        if result.returncode == 0:
            logger.info("Successfully synced evals to S3")
        else:
            logger.warning(f"Failed to sync evals: {result.stderr}")

    sync_thread = threading.Thread(target=sync_with_logging)
    sync_thread.daemon = True
    sync_thread.start()
    logger.info("Started async eval sync")


def sync_evals_blocking(eval_dir: str, logger: logging.Logger) -> bool:
    """Blocking sync of evals to S3. Returns success status."""
    os.makedirs("sevals", exist_ok=True)
    cp_result = subprocess.run(
        ['bash', '-c', f'cp -r {eval_dir}/* sevals/.'],
        capture_output=True, text=True
    )
    if cp_result.returncode != 0:
        logger.warning(f"Failed to copy evals: {cp_result.stderr}")
        return False
    logger.info("Final sync: syncing evals to S3...")
    result = _sd('sync', 'sevals')
    if result.returncode == 0:
        logger.info("Successfully synced evals to S3")
        return True
    else:
        logger.warning(f"Failed to sync evals: {result.stderr}")
        return False
