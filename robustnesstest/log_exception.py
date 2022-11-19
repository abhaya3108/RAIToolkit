#!/usr/bin/env python
import sys

def log_details(e):
    ''' This function return exception details (i.e. type, filename and line number) '''
    
    exception_type, exception_object, exception_traceback = sys.exc_info()
    filename = exception_traceback.tb_frame.f_code.co_filename
    line_number = exception_traceback.tb_lineno
    
    return exception_type, filename, line_number