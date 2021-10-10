#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: Metrics.py
@author: ImKe at 2021/5/10
@email: thq415_ic@yeah.net
@feature: #Enter features here
"""

from abc import abstractmethod


class Metrics:
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass