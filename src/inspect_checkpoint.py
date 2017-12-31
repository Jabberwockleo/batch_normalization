#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: inspect_checkpoint.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""
import sys
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

if __name__ == "__main__":
    print_tensors_in_checkpoint_file(sys.argv[1], None, True)
    pass
