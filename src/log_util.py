# -*- coding: utf-8 -*-
"""
로그 유틸리티. import한 "스크립트명.log" 파일에 로그 출력.
사용법:
import util.log_util as log
# log.change_file_mode("a") # 기본 모드는 "w"(덮어쓰기) 이지만 "a"(추가)로 변경가능.
log.info("로그 메시지 출력")
"""

import os
import logging
from datetime import datetime, timezone, timedelta


class CustomFormatter(logging.Formatter):
    """한국 시간대(UTC+9)로 로그 포맷을 설정하는 커스텀 포매터"""
    def format(self, record):
        # UTC+9 서울 타임존 설정
        kst = timezone(timedelta(hours=9))
        timestamp = datetime.fromtimestamp(record.created, tz=kst).strftime("%y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [INFO] {record.getMessage()}"


def setup_logger(script_file_path=None, file_mode="w"):
    """
    로거를 설정하고 반환합니다.
    
    Args:
        script_file_path (str, optional): 스크립트 파일 경로. None이면 현재 실행중인 스크립트 경로 사용
        file_mode (str): 파일 핸들러 모드 ("w", "a" 등)
    
    Returns:
        logging.Logger: 설정된 로거 객체
    """
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러가 있다면 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    
    # 파일 핸들러 (스크립트명.log로 저장)
    if script_file_path is None:
        import sys
        script_file_path = sys.argv[0] if sys.argv[0] else __file__
    
    script_name = os.path.splitext(os.path.basename(script_file_path))[0]
    script_dir = os.path.dirname(os.path.abspath(script_file_path))
    log_file_path = os.path.join(script_dir, f"{script_name}.log")
    
    file_handler = logging.FileHandler(log_file_path, mode=file_mode, encoding='utf-8')
    file_handler.setFormatter(CustomFormatter())
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger


# 모듈 로드 시 자동으로 로거 초기화 (스크립트 파일 경로 자동 감지, 기본 모드 "w")
_logger = setup_logger()
_current_script_path = None
_current_file_mode = "w"

# 모듈 레벨에서 로그 메서드들을 직접 노출
info = _logger.info
debug = _logger.debug
warning = _logger.warning
error = _logger.error
critical = _logger.critical

# 로거 객체도 직접 노출 (필요시 사용)
logger = _logger


def change_file_mode(file_mode):
    """
    파일 핸들러의 모드를 변경합니다.
    
    Args:
        file_mode (str): 새로운 파일 모드 ("w", "a" 등)
    """
    global _logger, info, debug, warning, error, critical, logger, _current_script_path, _current_file_mode
    
    _current_file_mode = file_mode
    
    # 현재 스크립트 경로가 설정되어 있으면 그것을 사용, 없으면 기본값 사용
    script_path = _current_script_path if _current_script_path else None
    
    _logger = setup_logger(script_path, file_mode)
    info = _logger.info
    debug = _logger.debug
    warning = _logger.warning
    error = _logger.error
    critical = _logger.critical
    logger = _logger 