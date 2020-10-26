#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod


class AbsDeepLearningInference(ABC):
    @abstractmethod
    def get_device(self):
        """
        get the real device to be invoked
        """

    @abstractmethod
    def quantification(self, dtype='int8'):
        """
        do quantification if chipset required
        """

    @abstractmethod
    def warm_up(self):
        """
        do load model and warm up
        """

    @abstractmethod
    def inference(self, input):
        """
        do model inference
        """

    @abstractmethod
    def test(self):
        """
        do inference test only
        """
