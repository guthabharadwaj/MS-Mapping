#!/usr/bin/env python3
import struct
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField

VELO_FIELDS = [
    PointField('x',         0, PointField.FLOAT32, 1),
    PointField('y',         4, PointField.FLOAT32, 1),
    PointField('z',         8, PointField.FLOAT32, 1),
    PointField('intensity', 12, PointField.FLOAT32, 1),
    PointField('time',      16, PointField.FLOAT32, 1),
    PointField('ring',      20, PointField.UINT16, 1),
]
VELO_POINT_STEP = 22  # 4*5 + 2

# Autoware-Universe Velodyne layout (32 bytes / pt):
#   x f32@0, y f32@4, z f32@8, intensity u8@12, return_type u8@13,
#   channel u16@14, azimuth f32@16, elevation f32@20, distance f32@24,
#   time_stamp u32@28 (relative ns)
AUTO_DTYPE = np.dtype({
    'names':   ['x', 'y', 'z', 'intensity', 'channel', 'time_stamp'],
    'formats': ['<f4', '<f4', '<f4', '<u1', '<u2', '<u4'],
    'offsets': [0, 4, 8, 12, 14, 28],
    'itemsize': 32,
})


def convert(msg):
    src = np.frombuffer(msg.data, dtype=AUTO_DTYPE, count=msg.width * msg.height)
    n = src.size
    out = bytearray(n * VELO_POINT_STEP)
    view = memoryview(out)
    # Pack into the velodyne_ros::Point layout.
    pos = 0
    xs = src['x'].astype(np.float32, copy=False)
    ys = src['y'].astype(np.float32, copy=False)
    zs = src['z'].astype(np.float32, copy=False)
    inten = src['intensity'].astype(np.float32)
    times = (src['time_stamp'] * 1e-9).astype(np.float32)
    rings = src['channel'].astype(np.uint16)
    block = np.zeros(n, dtype=np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('intensity', '<f4'), ('time', '<f4'), ('ring', '<u2'),
    ]))
    block['x'] = xs
    block['y'] = ys
    block['z'] = zs
    block['intensity'] = inten
    block['time'] = times
    block['ring'] = rings
    out_msg = PointCloud2()
    out_msg.header = msg.header
    out_msg.height = 1
    out_msg.width = n
    out_msg.fields = VELO_FIELDS
    out_msg.is_bigendian = False
    out_msg.point_step = VELO_POINT_STEP
    out_msg.row_step = VELO_POINT_STEP * n
    out_msg.is_dense = msg.is_dense
    out_msg.data = block.tobytes()
    pub.publish(out_msg)


if __name__ == '__main__':
    rospy.init_node('autoware_to_velodyne')
    in_topic = rospy.get_param('~input', '/sensing/lidar/pointcloud_raw_ex')
    out_topic = rospy.get_param('~output', '/velodyne_points')
    pub = rospy.Publisher(out_topic, PointCloud2, queue_size=10)
    rospy.Subscriber(in_topic, PointCloud2, convert, queue_size=10)
    rospy.loginfo('autoware_to_velodyne: %s -> %s', in_topic, out_topic)
    rospy.spin()
