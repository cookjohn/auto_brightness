#!/usr/bin/env python3
"""
自动亮度调节程序
通过笔记本电脑摄像头感知环境亮度，自动调节屏幕亮度

使用方法:
    python auto_brightness.py           # 持续运行模式
    python auto_brightness.py --once    # 单次调节
    python auto_brightness.py --preview # 预览模式（显示摄像头画面）
    python auto_brightness.py --learn   # 学习模式（校准环境光范围）
"""

import cv2
import numpy as np
import time
import argparse
import sys
import json
import os

try:
    import screen_brightness_control as sbc
except ImportError:
    print("请安装 screen_brightness_control: pip install screen-brightness-control")
    sys.exit(1)

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auto_brightness_config.json')

# 默认配置
DEFAULT_CONFIG = {
    'env_min': 0,        # 感应到的最小环境亮度
    'env_max': 255,      # 感应到的最大环境亮度
    'brightness_ratio': 1.0,  # 亮度适配比值（屏幕亮度 = 感应亮度 * ratio）
    'curve_gamma': 0.5,  # 曲线调整参数（用于平方根曲线）
    'interval': 2.0,     # 持续运行时的调节间隔（秒）
}


def load_config():
    """加载配置文件"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 合并默认配置（确保新字段有默认值）
                return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            print(f"加载配置失败: {e}，使用默认配置")
    return DEFAULT_CONFIG.copy()


def save_config(config):
    """保存配置文件"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"保存配置失败: {e}")
        return False


class AutoBrightness:
    def __init__(self, camera_id=0, smoothing=0.3, min_brightness=10, max_brightness=100):
        """
        初始化自动亮度控制器

        参数:
            camera_id: 摄像头ID，默认为0（通常是内置摄像头）
            smoothing: 平滑系数，0-1之间，越大变化越平滑
            min_brightness: 最小屏幕亮度（百分比）
            max_brightness: 最大屏幕亮度（百分比）
        """
        self.camera_id = camera_id
        self.smoothing = smoothing
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.last_brightness = None
        self.cap = None

        # 加载配置
        self.config = load_config()
        self.env_min = self.config['env_min']
        self.env_max = self.config['env_max']
        self.brightness_ratio = self.config['brightness_ratio']
        self.curve_gamma = self.config['curve_gamma']
        self.interval = self.config['interval']
    
    def open_camera(self):
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}")
        
        # 设置较低的分辨率以提高性能
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        return True
    
    def close_camera(self):
        """关闭摄像头"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def capture_frame(self, retry_on_dark=True):
        """
        捕获一帧图像

        参数:
            retry_on_dark: 当帧全黑时是否尝试重新打开摄像头
        """
        if self.cap is None:
            self.open_camera()

        if self.cap is None:
            raise RuntimeError("无法打开摄像头")

        ret, frame = self.cap.read()

        # 如果读取失败，尝试重新打开摄像头
        if not ret or frame is None:
            print("摄像头读取失败，尝试重新连接...")
            self.close_camera()
            time.sleep(0.5)
            try:
                self.open_camera()
                if self.cap is None:
                    raise RuntimeError("摄像头重连失败")
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise RuntimeError("无法从摄像头读取图像")
            except Exception as e:
                raise RuntimeError(f"摄像头重连失败: {e}")

        # 检测是否是全黑帧（可能是摄像头被禁用或遮挡）
        if retry_on_dark:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.max(gray) < 5:  # 几乎全黑
                print("检测到全黑帧，尝试重新初始化摄像头...")
                self.close_camera()
                time.sleep(1.0)
                self.open_camera()
                if self.cap is None:
                    raise RuntimeError("摄像头重新初始化失败")
                # 丢弃前几帧，让摄像头稳定
                for _ in range(5):
                    self.cap.read()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise RuntimeError("摄像头重新初始化后仍无法读取")

        return frame
    
    def analyze_brightness(self, frame):
        """
        分析图像亮度
        
        返回:
            0-255 之间的亮度值
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算平均亮度
        mean_brightness = np.mean(gray)
        
        # 可选：使用加权平均，中心区域权重更高
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_brightness = np.mean(center_region)
        
        # 综合考虑整体和中心区域
        weighted_brightness = 0.4 * mean_brightness + 0.6 * center_brightness
        
        return weighted_brightness
    
    def brightness_to_screen_level(self, env_brightness):
        """
        将环境亮度转换为屏幕亮度

        使用学习到的环境亮度范围 (env_min ~ env_max) 进行归一化
        然后应用亮度比值进行调整

        环境越亮 → 屏幕亮度越高
        环境越暗 → 屏幕亮度越低
        """
        # 根据学习到的范围进行归一化
        env_range = self.env_max - self.env_min
        if env_range <= 0:
            env_range = 255  # 防止除零

        # 将环境亮度限制在学习范围内，然后归一化到 0-1
        clamped = max(self.env_min, min(self.env_max, env_brightness))
        normalized = (clamped - self.env_min) / env_range

        # 应用曲线调整（gamma曲线）
        adjusted = np.power(normalized, self.curve_gamma)

        # 应用亮度比值
        adjusted = adjusted * self.brightness_ratio
        adjusted = max(0, min(1, adjusted))  # 限制在 0-1 范围内

        # 映射到屏幕亮度范围
        screen_brightness = self.min_brightness + adjusted * (self.max_brightness - self.min_brightness)

        return int(screen_brightness)
    
    def set_screen_brightness(self, target_brightness):
        """设置屏幕亮度，带平滑过渡"""
        if self.last_brightness is None:
            self.last_brightness = target_brightness
        
        # 平滑过渡
        smoothed = self.last_brightness + (target_brightness - self.last_brightness) * (1 - self.smoothing)
        smoothed = int(smoothed)
        
        # 限制范围
        smoothed = max(self.min_brightness, min(self.max_brightness, smoothed))
        
        try:
            sbc.set_brightness(smoothed)
            self.last_brightness = smoothed
            return smoothed
        except Exception as e:
            print(f"设置亮度失败: {e}")
            return None
    
    def get_current_screen_brightness(self):
        """获取当前屏幕亮度"""
        try:
            brightness = sbc.get_brightness()
            if isinstance(brightness, list):
                return brightness[0]
            return brightness
        except Exception as e:
            print(f"获取亮度失败: {e}")
            return None
    
    def adjust_once(self, verbose=True, step_threshold=10):
        """
        单次亮度调节

        参数:
            verbose: 是否显示详细信息
            step_threshold: 亮度变化阈值（%），只有变化超过此值才调整
        """
        try:
            # 打开摄像头检测
            self.open_camera()
            frame = self.capture_frame()
            self.close_camera()  # 检测完立即关闭

            env_brightness = self.analyze_brightness(frame)
            target = self.brightness_to_screen_level(env_brightness)
            current = self.get_current_screen_brightness() or 50

            # 只有当变化超过阈值时才调整
            diff = abs(target - current)
            if diff >= step_threshold:
                actual = self.set_screen_brightness(target)
                if verbose:
                    print(f"环境亮度: {env_brightness:.1f}/255 → 屏幕亮度: {current}% → {actual}%")
                return actual
            else:
                if verbose:
                    print(f"环境亮度: {env_brightness:.1f}/255 | 当前: {current}% (变化{diff}%<{step_threshold}%，跳过)")
                return current

        except Exception as e:
            print(f"调节失败: {e}")
            self.close_camera()  # 确保关闭摄像头
            return None
    
    def run_continuous(self, interval=None, verbose=True):
        """
        持续运行模式

        参数:
            interval: 调节间隔（秒），如果为None则使用配置中的值
            verbose: 是否显示详细信息
        """
        if interval is not None:
            self.interval = interval

        print("自动亮度调节已启动")
        print(f"   调节间隔: {self.interval}秒")
        print(f"   亮度范围: {self.min_brightness}% - {self.max_brightness}%")
        print(f"   变化阈值: 10%")
        print("   按 Ctrl+C 停止\n")

        try:
            while True:
                self.adjust_once(verbose)
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n\n已停止自动亮度调节")
    
    def save_current_config(self):
        """保存当前配置到文件"""
        self.config['env_min'] = self.env_min
        self.config['env_max'] = self.env_max
        self.config['brightness_ratio'] = self.brightness_ratio
        self.config['curve_gamma'] = self.curve_gamma
        self.config['interval'] = self.interval
        return save_config(self.config)

    def set_interval(self, interval):
        """设置调节间隔"""
        self.interval = max(0.5, min(60.0, interval))
        print(f"调节间隔已设置为: {self.interval:.1f}秒")
        self.save_current_config()

    def set_brightness_ratio(self, ratio):
        """设置亮度适配比值"""
        self.brightness_ratio = max(0.1, min(3.0, ratio))
        print(f"亮度比值已设置为: {self.brightness_ratio:.2f}")
        self.save_current_config()

    def run_learn(self):
        """
        学习模式：校准环境光的最大/最小亮度范围

        操作说明:
        - 按 'l' 记录当前为最低亮度（在最暗环境下按）
        - 按 'h' 记录当前为最高亮度（在最亮环境下按）
        - 按 '+/-' 调整亮度比值
        - 按 's' 保存配置
        - 按 'r' 重置为默认配置
        - 按 'q' 退出
        """
        print("=" * 50)
        print("学习模式 - 环境光校准")
        print("=" * 50)
        print("\n操作说明:")
        print("  'l' - 记录当前环境为 最低亮度")
        print("  'h' - 记录当前环境为 最高亮度")
        print("  '+' - 增加亮度比值 (+0.1)")
        print("  '-' - 减少亮度比值 (-0.1)")
        print("  's' - 保存配置")
        print("  'r' - 重置为默认配置")
        print("  'q' - 退出学习模式")
        print("\n当前配置:")
        print(f"  环境亮度范围: {self.env_min:.1f} ~ {self.env_max:.1f}")
        print(f"  亮度比值: {self.brightness_ratio:.2f}")
        print("-" * 50 + "\n")

        samples_low = []
        samples_high = []

        try:
            self.open_camera()

            while True:
                frame = self.capture_frame()
                env_brightness = self.analyze_brightness(frame)
                target = self.brightness_to_screen_level(env_brightness)
                current = self.get_current_screen_brightness()

                # 显示信息
                info_lines = [
                    f"Env Brightness: {env_brightness:.1f}",
                    f"Learned Range: {self.env_min:.1f} ~ {self.env_max:.1f}",
                    f"Ratio: {self.brightness_ratio:.2f}",
                    f"Target: {target}%  Current: {current}%",
                ]

                # 绘制亮度条
                bar_width = int((env_brightness / 255) * 300)
                cv2.rectangle(frame, (10, 10), (310, 30), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 10), (10 + bar_width, 30), (0, 255, 255), -1)

                # 绘制学习范围标记
                if self.env_max > self.env_min:
                    min_pos = int((self.env_min / 255) * 300) + 10
                    max_pos = int((self.env_max / 255) * 300) + 10
                    cv2.line(frame, (min_pos, 5), (min_pos, 35), (255, 0, 0), 2)
                    cv2.line(frame, (max_pos, 5), (max_pos, 35), (0, 0, 255), 2)

                for i, text in enumerate(info_lines):
                    cv2.putText(frame, text, (10, 55 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(frame, "L=low H=high +/-=ratio S=save Q=quit", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                cv2.imshow('Learn Mode - Brightness Calibration', frame)

                key = cv2.waitKey(100) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('l'):
                    # 记录最低亮度样本
                    samples_low.append(env_brightness)
                    if len(samples_low) >= 3:
                        self.env_min = np.mean(samples_low[-10:])  # 取最近10个样本的平均
                    else:
                        self.env_min = env_brightness
                    print(f"已记录最低亮度: {self.env_min:.1f} (样本数: {len(samples_low)})")
                elif key == ord('h'):
                    # 记录最高亮度样本
                    samples_high.append(env_brightness)
                    if len(samples_high) >= 3:
                        self.env_max = np.mean(samples_high[-10:])
                    else:
                        self.env_max = env_brightness
                    print(f"已记录最高亮度: {self.env_max:.1f} (样本数: {len(samples_high)})")
                elif key == ord('+') or key == ord('='):
                    self.brightness_ratio = min(3.0, self.brightness_ratio + 0.1)
                    print(f"亮度比值: {self.brightness_ratio:.2f}")
                elif key == ord('-'):
                    self.brightness_ratio = max(0.1, self.brightness_ratio - 0.1)
                    print(f"亮度比值: {self.brightness_ratio:.2f}")
                elif key == ord('s'):
                    if self.save_current_config():
                        print("配置已保存!")
                        print(f"  环境亮度范围: {self.env_min:.1f} ~ {self.env_max:.1f}")
                        print(f"  亮度比值: {self.brightness_ratio:.2f}")
                elif key == ord('r'):
                    self.env_min = DEFAULT_CONFIG['env_min']
                    self.env_max = DEFAULT_CONFIG['env_max']
                    self.brightness_ratio = DEFAULT_CONFIG['brightness_ratio']
                    samples_low.clear()
                    samples_high.clear()
                    print("已重置为默认配置")
                elif key == ord('a'):
                    actual = self.set_screen_brightness(target)
                    print(f"已应用亮度: {actual}%")

        finally:
            self.close_camera()
            cv2.destroyAllWindows()

        print("\n学习模式结束")

    def run_preview(self):
        """预览模式：显示摄像头画面和亮度信息"""
        print("预览模式 - 按 'q' 退出, 按 'a' 应用当前亮度\n")
        
        try:
            self.open_camera()
            
            while True:
                frame = self.capture_frame()
                env_brightness = self.analyze_brightness(frame)
                target = self.brightness_to_screen_level(env_brightness)
                current = self.get_current_screen_brightness()
                
                # 在画面上显示信息
                info_text = [
                    f"Environment: {env_brightness:.1f}/255",
                    f"Target Brightness: {target}%",
                    f"Current Brightness: {current}%"
                ]
                
                # 绘制亮度条
                bar_width = int((env_brightness / 255) * 300)
                cv2.rectangle(frame, (10, 10), (310, 30), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 10), (10 + bar_width, 30), (0, 255, 255), -1)
                
                # 显示文字
                for i, text in enumerate(info_text):
                    cv2.putText(frame, text, (10, 60 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Auto Brightness Preview', frame)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    actual = self.set_screen_brightness(target)
                    print(f"已应用亮度: {actual}%")
            
        finally:
            self.close_camera()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='通过摄像头感知环境亮度自动调节屏幕亮度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python auto_brightness.py                    # 持续运行
  python auto_brightness.py --once             # 单次调节
  python auto_brightness.py --preview          # 预览模式
  python auto_brightness.py --learn            # 学习模式（校准环境光范围）
  python auto_brightness.py --ratio 1.2        # 设置亮度比值
  python auto_brightness.py --show-config      # 显示当前配置
  python auto_brightness.py --interval 5       # 每5秒调节一次
  python auto_brightness.py --min 20 --max 80  # 限制亮度范围
        '''
    )

    parser.add_argument('--once', action='store_true',
                        help='仅调节一次后退出')
    parser.add_argument('--preview', action='store_true',
                        help='预览模式（显示摄像头画面）')
    parser.add_argument('--learn', action='store_true',
                        help='学习模式（校准环境光的最大/最小亮度）')
    parser.add_argument('--ratio', type=float, default=None,
                        help='设置亮度适配比值（0.1-3.0），控制感应亮度与屏幕亮度的映射关系')
    parser.add_argument('--show-config', action='store_true',
                        help='显示当前配置并退出')
    parser.add_argument('--reset-config', action='store_true',
                        help='重置配置为默认值')
    parser.add_argument('--interval', type=float, default=None,
                        help='设置调节间隔（秒），会保存到配置文件')
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头ID，默认0')
    parser.add_argument('--min', type=int, default=10,
                        help='最小屏幕亮度（%%），默认10')
    parser.add_argument('--max', type=int, default=100,
                        help='最大屏幕亮度（%%），默认100')
    parser.add_argument('--smooth', type=float, default=0.3,
                        help='平滑系数（0-1），默认0.3')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式，减少输出')

    args = parser.parse_args()

    # 显示配置
    if args.show_config:
        config = load_config()
        print("当前配置:")
        print(f"  环境亮度范围: {config['env_min']:.1f} ~ {config['env_max']:.1f}")
        print(f"  亮度适配比值: {config['brightness_ratio']:.2f}")
        print(f"  调节间隔: {config['interval']:.1f}秒")
        print(f"  曲线Gamma: {config['curve_gamma']:.2f}")
        print(f"\n配置文件: {CONFIG_FILE}")
        return

    # 重置配置
    if args.reset_config:
        save_config(DEFAULT_CONFIG)
        print("配置已重置为默认值")
        return

    # 创建控制器
    controller = AutoBrightness(
        camera_id=args.camera,
        smoothing=args.smooth,
        min_brightness=args.min,
        max_brightness=args.max
    )

    # 设置亮度比值
    if args.ratio is not None:
        controller.set_brightness_ratio(args.ratio)

    # 设置调节间隔
    if args.interval is not None:
        controller.set_interval(args.interval)

    # 如果只是设置参数（没有指定运行模式），则退出
    if (args.ratio is not None or args.interval is not None) and \
       not (args.once or args.preview or args.learn):
        return

    # 运行对应模式
    if args.learn:
        controller.run_learn()
    elif args.once:
        controller.adjust_once(verbose=not args.quiet)
    elif args.preview:
        controller.run_preview()
    else:
        controller.run_continuous(verbose=not args.quiet)


if __name__ == '__main__':
    main()
