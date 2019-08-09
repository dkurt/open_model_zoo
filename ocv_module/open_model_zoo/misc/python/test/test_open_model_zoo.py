#!/usr/bin/env python
import os
import numpy as np

import cv2 as cv
import cv2.open_model_zoo as omz

from tests_common import NewOpenCVTests, unittest

class open_model_zoo_test(NewOpenCVTests):

    def setUp(self):
        super(open_model_zoo_test, self).setUp()

    def test_downloadGoogleDriveWithConfirm(self):
        t = omz.topologies.densenet_161()
        modelPath = t.getModelPath()
        configPath = t.getConfigPath()

        self.assertEqual(os.path.basename(modelPath), 'densenet-161.caffemodel')
        self.assertEqual(os.path.basename(configPath), 'densenet-161.prototxt')
        os.remove(modelPath)
        os.remove(configPath)

    def test_convertToIR(self):
        t = omz.topologies.squeezenet1_0()
        xmlPath, binPath = t.convertToIR()
        self.assertEqual(os.path.basename(xmlPath), 'squeezenet1_0.xml')
        self.assertEqual(os.path.basename(binPath), 'squeezenet1_0.bin')
        os.remove(xmlPath)
        os.remove(binPath)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
