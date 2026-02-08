"""
Colored Logger Module

Provides colorized logging output for better console readability.
"""

import logging
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
        
        # Custom colors for specific messages
        'SUCCESS': '\033[92m',    # Bright Green
        'STAGE': '\033[96m',      # Bright Cyan
        'VALIDATION': '\033[93m', # Bright Yellow
        'GENERATION': '\033[94m', # Bright Blue
        'MODEL': '\033[95m',      # Bright Magenta
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get the original formatted message
        log_message = super().format(record)
        
        # Add color based on log level
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        
        # Special handling for custom prefixes
        if hasattr(record, 'color_type'):
            color = self.COLORS.get(record.color_type.upper(), level_color)
            return f"{color}{log_message}{self.COLORS['RESET']}"
        
        return f"{level_color}{log_message}{self.COLORS['RESET']}"


class ColoredLogger:
    """Wrapper for logging with color support."""
    
    def __init__(self, name: str = 'root', level: int = logging.INFO):
        """
        Initialize colored logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
    @staticmethod
    def setup_colored_logging(logger: Optional[logging.Logger] = None) -> logging.Logger:
        """
        Setup colored logging for a logger.
        
        Args:
            logger: Logger instance (if None, uses root logger)
            
        Returns:
            Configured logger
        """
        if logger is None:
            logger = logging.getLogger()
        
        # Remove existing handlers
        logger.handlers = []
        
        # Create console handler with colored formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Use colored formatter
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    @staticmethod
    def log_success(message: str, logger: Optional[logging.Logger] = None):
        """Log a success message in bright green."""
        if logger is None:
            logger = logging.getLogger()
        record = logger.makeRecord(
            logger.name, logging.INFO, '', 0, message, (), None
        )
        record.color_type = 'SUCCESS'
        logger.handle(record)
    
    @staticmethod
    def log_stage(message: str, logger: Optional[logging.Logger] = None):
        """Log a stage message in bright cyan."""
        if logger is None:
            logger = logging.getLogger()
        record = logger.makeRecord(
            logger.name, logging.INFO, '', 0, message, (), None
        )
        record.color_type = 'STAGE'
        logger.handle(record)
    
    @staticmethod
    def log_validation(message: str, logger: Optional[logging.Logger] = None):
        """Log a validation message in bright yellow."""
        if logger is None:
            logger = logging.getLogger()
        record = logger.makeRecord(
            logger.name, logging.WARNING, '', 0, message, (), None
        )
        record.color_type = 'VALIDATION'
        logger.handle(record)
    
    @staticmethod
    def log_generation(message: str, logger: Optional[logging.Logger] = None):
        """Log a generation message in bright blue."""
        if logger is None:
            logger = logging.getLogger()
        record = logger.makeRecord(
            logger.name, logging.INFO, '', 0, message, (), None
        )
        record.color_type = 'GENERATION'
        logger.handle(record)
    
    @staticmethod
    def log_model(message: str, logger: Optional[logging.Logger] = None):
        """Log a model-related message in bright magenta."""
        if logger is None:
            logger = logging.getLogger()
        record = logger.makeRecord(
            logger.name, logging.INFO, '', 0, message, (), None
        )
        record.color_type = 'MODEL'
        logger.handle(record)


def print_colored(message: str, color: str = 'INFO'):
    """
    Print a colored message directly to console.
    
    Args:
        message: Message to print
        color: Color name (INFO, WARNING, ERROR, SUCCESS, STAGE, etc.)
    """
    colors = ColoredFormatter.COLORS
    color_code = colors.get(color.upper(), colors['RESET'])
    print(f"{color_code}{message}{colors['RESET']}")
