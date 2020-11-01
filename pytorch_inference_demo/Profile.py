#!/usr/bin/env python
# coding: utf-8


import cProfile
import pstats
from io import StringIO
import logging


profile_enabled = False


def enable_profile():
    globals()['profile_enabled'] = True


def profile(fun):
    def _profile(*args, **kwargs):
        if globals().get('profile_enabled'):
            pr  = cProfile.Profile()
            pr.enable()
            r = fun(*args, **kwargs)
            pr.disable()
            s = StringIO()
            pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats()
            logging.info(s.getvalue())
        else:
            r = fun(*args, **kwargs)
        return r

    return _profile


if __name__ == '__main__':
    import time
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    enable_profile()

    @profile
    def test(s):
        time.sleep(s)

    test(1)
