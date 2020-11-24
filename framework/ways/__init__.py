"""
lists the names you will need and what you can import with:
>>> from framework.ways import
"""

from .streets_map import Junction, StreetsMap, Link, ROAD_SPEEDS, MIN_ROAD_SPEED, MAX_ROAD_SPEED, kmph_to_mpm

__all__ = ['Junction', 'Link', 'StreetsMap',
           'ROAD_SPEEDS', 'MIN_ROAD_SPEED', 'MAX_ROAD_SPEED', 'kmph_to_mpm']
