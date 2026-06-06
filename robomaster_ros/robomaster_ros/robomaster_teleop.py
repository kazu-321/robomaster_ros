from __future__ import annotations

import math
import time
import threading
from typing import Any, Optional, Sequence

import geometry_msgs.msg
import nav_msgs.msg
import rclpy
import rclpy.node
import robomaster_msgs.msg
import sensor_msgs.msg


def clamp_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return value


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_to_yaw(orientation: geometry_msgs.msg.Quaternion) -> float:
    siny_cosp = 2.0 * (
        orientation.w * orientation.z + orientation.x * orientation.y
    )
    cosy_cosp = 1.0 - 2.0 * (
        orientation.y * orientation.y + orientation.z * orientation.z
    )
    return math.atan2(siny_cosp, cosy_cosp)


class RoboMasterTeleop(rclpy.node.Node):
    def __init__(self) -> None:
        super().__init__('robomaster_teleop')

        self.joy_topic = self.declare_parameter('joy_topic', 'joy').value
        self.odom_topic = self.declare_parameter('odom_topic', 'odom').value
        self.joint_states_topic = self.declare_parameter(
            'joint_states_topic', 'joint_states').value
        self.joint_states_partial_topic = self.declare_parameter(
            'joint_states_partial_topic', 'joint_states_p').value
        self.cmd_vel_topic = self.declare_parameter('cmd_vel_topic', 'cmd_vel').value
        self.cmd_gimbal_topic = self.declare_parameter('cmd_gimbal_topic', 'cmd_gimbal').value
        self.gimbal_joint_name = self.declare_parameter('gimbal_joint_name', 'gimbal_joint').value

        self.left_x_axis = int(self.declare_parameter('left_x_axis', 0).value)
        self.left_y_axis = int(self.declare_parameter('left_y_axis', 1).value)
        self.right_x_axis = int(self.declare_parameter('right_x_axis', 3).value)
        self.right_y_axis = int(self.declare_parameter('right_y_axis', 4).value)
        self.l1_button = int(self.declare_parameter('l1_button', 4).value)
        self.r1_button = int(self.declare_parameter('r1_button', 5).value)

        self.linear_x_scale = float(self.declare_parameter('linear_x_scale', -1.5).value)
        self.linear_y_scale = float(self.declare_parameter('linear_y_scale', -1.5).value)
        self.gimbal_yaw_speed_deg_s = float(
            self.declare_parameter('gimbal_yaw_speed_deg_s', -180.0).value)
        self.gimbal_pitch_speed_deg_s = float(
            self.declare_parameter('gimbal_pitch_speed_deg_s', -180.0).value)
        self.max_gimbal_yaw_deg = float(
            self.declare_parameter('max_gimbal_yaw_deg', 200.0).value)
        self.god_mode_yaw_speed_deg_s = float(
            self.declare_parameter('god_mode_yaw_speed_deg_s', 400.0).value)
        self.follow_gimbal_yaw_kp = float(
            self.declare_parameter('follow_gimbal_yaw_kp', 4.0).value)
        self.blaster_hz = float(self.declare_parameter('blaster_hz', 8.0).value)
        self.blaster_sound_id = int(self.declare_parameter('blaster_sound_id', 15).value)
        self.publish_rate_hz = float(self.declare_parameter('publish_rate_hz', 20.0).value)
        self.joy_timeout_sec = float(self.declare_parameter('joy_timeout_sec', 0.25).value)
        self.deadzone = float(self.declare_parameter('deadzone', 0.1).value)

        self._max_gimbal_yaw_rad = math.radians(self.max_gimbal_yaw_deg)
        self._god_mode_yaw_speed_rad_s = math.radians(self.god_mode_yaw_speed_deg_s)
        self._gimbal_yaw_speed_rad_s = math.radians(self.gimbal_yaw_speed_deg_s)
        self._gimbal_pitch_speed_rad_s = math.radians(self.gimbal_pitch_speed_deg_s)

        self.cmd_vel_pub = self.create_publisher(geometry_msgs.msg.Twist, self.cmd_vel_topic, 10)
        self.cmd_gimbal_pub = self.create_publisher(
            robomaster_msgs.msg.GimbalCommand, self.cmd_gimbal_topic, 10)
        self.blaster_led_pub = self.create_publisher(
            robomaster_msgs.msg.BlasterLED, 'blaster_led', 10)
        self.cmd_sound_pub = self.create_publisher(
            robomaster_msgs.msg.SpeakerCommand, 'cmd_sound', 10)

        self.create_subscription(sensor_msgs.msg.Joy, self.joy_topic, self.joy_callback, 10)
        self.create_subscription(nav_msgs.msg.Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(
            sensor_msgs.msg.JointState, self.joint_states_topic, self.joint_states_callback, 10)
        self.create_subscription(
            sensor_msgs.msg.JointState, self.joint_states_partial_topic,
            self.joint_states_callback, 10)

        self._timer = self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self.publish_loop)

        self._lock = threading.Lock()
        self._joy: Optional[sensor_msgs.msg.Joy] = None
        self._last_joy_time = self.get_clock().now()
        self._l1_was_pressed = False
        self._god_mode = 0

        self._chassis_yaw_raw: Optional[float] = None
        self._chassis_yaw_continuous: Optional[float] = None
        self._gimbal_yaw_raw: Optional[float] = None
        self._gimbal_yaw_continuous: Optional[float] = None
        self._relative_yaw_raw: Optional[float] = None
        self._relative_yaw_continuous: Optional[float] = None
        self._blaster_on = False
        self._blaster_pulse_ns = int(1e9 / max(self.blaster_hz, 0.1))
        self._blaster_stop = False
        self._blaster_thread = threading.Thread(
            target=self._blaster_worker, name='robomaster_blaster_worker', daemon=True)
        self._blaster_thread.start()

    @staticmethod
    def _axis(axes: Sequence[float], index: int) -> float:
        if index < 0 or index >= len(axes):
            return 0.0
        return float(axes[index])

    @staticmethod
    def _button(buttons: Sequence[int], index: int) -> bool:
        if index < 0 or index >= len(buttons):
            return False
        return bool(buttons[index])

    def _unwrap(self, previous_raw: Optional[float], previous_continuous: Optional[float],
                new_raw: float) -> float:
        if previous_raw is None or previous_continuous is None:
            return new_raw
        return previous_continuous + normalize_angle(new_raw - previous_raw)

    def _joint_state_gimbal_yaw(self, msg: sensor_msgs.msg.JointState) -> Optional[float]:
        if not msg.name or not msg.position:
            return None
        for index, name in enumerate(msg.name):
            if name == self.gimbal_joint_name or name.endswith(f'/{self.gimbal_joint_name}'):
                if index < len(msg.position):
                    return float(msg.position[index])
        return None

    def joy_callback(self, msg: sensor_msgs.msg.Joy) -> None:
        with self._lock:
            self._joy = msg
            self._last_joy_time = self.get_clock().now()

    def joint_states_callback(self, msg: sensor_msgs.msg.JointState) -> None:
        yaw = self._joint_state_gimbal_yaw(msg)
        if yaw is None:
            return
        with self._lock:
            self._gimbal_yaw_continuous = self._unwrap(
                self._gimbal_yaw_raw, self._gimbal_yaw_continuous, yaw)
            self._gimbal_yaw_raw = yaw
            self._update_relative_yaw_locked()

    def odom_callback(self, msg: nav_msgs.msg.Odometry) -> None:
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        with self._lock:
            self._chassis_yaw_continuous = self._unwrap(
                self._chassis_yaw_raw, self._chassis_yaw_continuous, yaw)
            self._chassis_yaw_raw = yaw
            self._update_relative_yaw_locked()

    def _update_relative_yaw_locked(self) -> None:
        if self._chassis_yaw_raw is None or self._gimbal_yaw_raw is None:
            return
        relative_raw = normalize_angle(self._chassis_yaw_raw - self._gimbal_yaw_raw)
        self._relative_yaw_continuous = self._unwrap(
            self._relative_yaw_raw, self._relative_yaw_continuous, relative_raw)
        self._relative_yaw_raw = relative_raw

    def _stale_joy(self) -> bool:
        return (self.get_clock().now() - self._last_joy_time).nanoseconds > int(
            self.joy_timeout_sec * 1e9)

    def _make_zero_twist(self) -> geometry_msgs.msg.Twist:
        return geometry_msgs.msg.Twist()

    def _make_zero_gimbal(self) -> robomaster_msgs.msg.GimbalCommand:
        return robomaster_msgs.msg.GimbalCommand()

    def _publish_blaster(self, brightness: float) -> None:
        self.blaster_led_pub.publish(robomaster_msgs.msg.BlasterLED(brightness=brightness))

    def _publish_sound(self) -> None:
        self.cmd_sound_pub.publish(
            robomaster_msgs.msg.SpeakerCommand(
                control=1,
                sound_id=self.blaster_sound_id,
                times=1,
            ))

    def _set_blaster_state(self, on: bool, play_sound: bool = False) -> None:
        self._blaster_on = on
        self._publish_blaster(1.0 if on else 0.0)
        if on and play_sound:
            self._publish_sound()

    def _blaster_worker(self) -> None:
        active = False
        next_toggle_ns: Optional[int] = None
        last_r1 = False
        pulse_ns = self._blaster_pulse_ns

        while not self._blaster_stop and rclpy.ok():
            with self._lock:
                joy = self._joy
                stale = self._stale_joy() if joy is not None else True
                r1_pressed = bool(joy is not None and not stale and self._button(joy.buttons, self.r1_button))

            now_ns = time.monotonic_ns()

            if joy is None or stale:
                if active or self._blaster_on:
                    active = False
                    next_toggle_ns = None
                    self._set_blaster_state(False, play_sound=False)
                last_r1 = False
                time.sleep(0.01)
                continue

            if r1_pressed and not last_r1:
                active = True
                self._set_blaster_state(True, play_sound=True)
                next_toggle_ns = now_ns + pulse_ns
            elif active and next_toggle_ns is not None and now_ns >= next_toggle_ns:
                if self._blaster_on:
                    self._set_blaster_state(False, play_sound=False)
                    if r1_pressed:
                        next_toggle_ns = now_ns + pulse_ns
                    else:
                        active = False
                        next_toggle_ns = None
                else:
                    if r1_pressed:
                        self._set_blaster_state(True, play_sound=True)
                        next_toggle_ns = now_ns + pulse_ns
                    else:
                        active = False
                        next_toggle_ns = None

            last_r1 = r1_pressed

            if next_toggle_ns is None:
                time.sleep(0.01)
            else:
                remaining_ns = max(0, next_toggle_ns - time.monotonic_ns())
                time.sleep(min(0.01, remaining_ns / 1e9))

    def _compute_cmd_vel(self, joy: sensor_msgs.msg.Joy) -> geometry_msgs.msg.Twist:
        twist = geometry_msgs.msg.Twist()
        left_x = clamp_deadzone(self._axis(joy.axes, self.left_x_axis), self.deadzone)
        left_y = clamp_deadzone(self._axis(joy.axes, self.left_y_axis), self.deadzone)
        twist.linear.x = -left_y * self.linear_x_scale
        twist.linear.y = -left_x * self.linear_y_scale

        l1_pressed = self._button(joy.buttons, self.l1_button)
        if self._gimbal_yaw_continuous is None:
            twist.angular.z = 0.0
            return twist

        yaw_error = self._gimbal_yaw_continuous
        if not l1_pressed:
            self._god_mode = 0
            self._l1_was_pressed = False
            twist.angular.z = max(
                -self._god_mode_yaw_speed_rad_s,
                min(self._god_mode_yaw_speed_rad_s,
                    self.follow_gimbal_yaw_kp * yaw_error))
            return twist

        if not self._l1_was_pressed:
            self._god_mode = 0

        self._l1_was_pressed = True
        if self._relative_yaw_continuous is None:
            twist.angular.z = 0.0
            return twist

        yaw_error = self._relative_yaw_continuous
        if self._god_mode == 0:
            if yaw_error >= self._max_gimbal_yaw_rad:
                self._god_mode = 1
                twist.angular.z = -self._god_mode_yaw_speed_rad_s
            else:
                twist.angular.z = self._god_mode_yaw_speed_rad_s
        else:
            if yaw_error <= -self._max_gimbal_yaw_rad:
                self._god_mode = 0
                twist.angular.z = self._god_mode_yaw_speed_rad_s
            else:
                twist.angular.z = -self._god_mode_yaw_speed_rad_s
        return twist

    def _compute_cmd_gimbal(self, joy: sensor_msgs.msg.Joy) -> robomaster_msgs.msg.GimbalCommand:
        cmd = robomaster_msgs.msg.GimbalCommand()
        right_x = clamp_deadzone(self._axis(joy.axes, self.right_x_axis), self.deadzone)
        right_y = clamp_deadzone(self._axis(joy.axes, self.right_y_axis), self.deadzone)
        cmd.yaw_speed = right_x * self._gimbal_yaw_speed_rad_s
        cmd.pitch_speed = right_y * self._gimbal_pitch_speed_rad_s
        return cmd

    def publish_loop(self) -> None:
        with self._lock:
            joy = self._joy
            stale = self._stale_joy() if joy is not None else True

            if joy is None or stale:
                self._god_mode = 0
                self._l1_was_pressed = False
                twist = self._make_zero_twist()
                gimbal = self._make_zero_gimbal()
            else:
                twist = self._compute_cmd_vel(joy)
                gimbal = self._compute_cmd_gimbal(joy)

        self.cmd_vel_pub.publish(twist)
        self.cmd_gimbal_pub.publish(gimbal)


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    node = RoboMasterTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._blaster_stop = True
        if node._blaster_thread.is_alive():
            node._blaster_thread.join(timeout=1.0)
        node._set_blaster_state(False, play_sound=False)
        node.destroy_node()
        rclpy.try_shutdown()
