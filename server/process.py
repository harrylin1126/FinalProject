import cv2
import mediapipe as mp
import math
import numpy as np
import time

def calculate_angle(a, b, c):
    """計算兩個向量之間的夾角"""
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))
    return angle

def is_right_elbow_above_shoulder(landmarks):
    """檢查右手肘是否高於肩膀"""
    return landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y < landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y

def check_right_elbow_angle(angle):
    """檢查右手肘角度是否在正確範圍內"""
    return 160 <= angle <= 170

def check_left_knee_angle(angle):
    """檢查左膝蓋角度是否接近伸直"""
    return angle >= 170

def check_foot_direction(landmarks):
    """檢查腳尖朝向"""
    left_foot = np.array([landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].y])
    right_foot = np.array([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y])
    foot_direction = left_foot - right_foot
    return foot_direction[1] > 0

def calculate_rotation_angle(landmarks, is_shoulder=True):
    """計算肩膀和髖部的旋轉角度"""
    if is_shoulder:
        left = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    else:
        left = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        right = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
    
    dx = right.x - left.x
    dy = right.y - left.y
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def analyze_pitcher_pose(video_path):
    """
    分析投手投球動作的主要函式
    
    Args:
        video_path (str): 影片檔案的路徑
    
    Returns:
        dict: 包含分析結果的字典
    """
    # 初始化 MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # 初始化變數
    elbow_phase_complete = False
    knee_phase_complete = False
    correct_elbow_angle_detected = False
    correct_knee_angle_detected = False
    correct_foot_direction_detected = False
    hip_rotates_faster = False
    detection_phase = "Not started"
    
    # 時間追蹤變數
    start_time = None
    is_tracking_left_arm = False
    
    prev_shoulder_angle = None
    prev_hip_angle = None
    elbow_angles = []
    
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Pitcher Pose Analysis', cv2.WINDOW_NORMAL)
    
    timing_data = {
        'movement_duration': None,
        'movement_too_slow': False
    }

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 左手臂時間追蹤
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                if left_wrist.y < left_shoulder.y and not is_tracking_left_arm:
                    start_time = time.time()
                    is_tracking_left_arm = True
                    #print("開始追蹤投球動作時間")
                
                elif is_tracking_left_arm and left_wrist.y > left_shoulder.y:
                    end_time = time.time()
                    duration = end_time - start_time
                    timing_data['movement_duration'] = duration
                    
                    if duration > 5.0:
                        timing_data['movement_too_slow'] = True
                        print(f"投球動作時間過長 ({duration:.2f} 秒)")
                        cv2.putText(frame, f"Movement too slow: {duration:.2f}s", 
                                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 0, 255), 2)
                    
                    is_tracking_left_arm = False
                    start_time = None

                # 計算右手肘角度
                right_elbow_angle = calculate_angle(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                    [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                    [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                )

                # 計算左膝蓋角度
                left_knee_angle = calculate_angle(
                    [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                    [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                    [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                )

                # 檢測邏輯
                if detection_phase == "Not started" and is_right_elbow_above_shoulder(landmarks):
                    detection_phase = "Elbow phase"
                    print("檢測開始：右手肘高於肩膀")
                    prev_shoulder_angle = calculate_rotation_angle(landmarks, True)
                    prev_hip_angle = calculate_rotation_angle(landmarks, False)
               
                elif detection_phase == "Elbow phase":
                    elbow_angles.append(right_elbow_angle)
                    if check_right_elbow_angle(right_elbow_angle):
                        correct_elbow_angle_detected = True
                   
                    if prev_shoulder_angle is not None and prev_hip_angle is not None:
                        shoulder_rotation = abs(calculate_rotation_angle(landmarks, True) - prev_shoulder_angle)
                        hip_rotation = abs(calculate_rotation_angle(landmarks, False) - prev_hip_angle)
                        if hip_rotation > shoulder_rotation:
                            hip_rotates_faster = True
                   
                    prev_shoulder_angle = calculate_rotation_angle(landmarks, True)
                    prev_hip_angle = calculate_rotation_angle(landmarks, False)
                   
                    '''if is_right_elbow_above_shoulder(landmarks) and right_elbow_angle > 170:
                        elbow_phase_complete = True
                        detection_phase = "Knee phase"
                        if elbow_angles:
                            avg_elbow_angle = sum(elbow_angles) / len(elbow_angles)
                            print(f"手肘檢測階段的平均角度：{avg_elbow_angle:.2f}度")'''

                elif detection_phase == "Knee phase":
                    if check_left_knee_angle(left_knee_angle):
                        correct_knee_angle_detected = True
                        knee_phase_complete = True
                        detection_phase = "Ended"
                        print("膝蓋階段結束")
                   
                    if check_foot_direction(landmarks):
                        correct_foot_direction_detected = True

                # 繪製骨架
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            cv2.imshow('Pitcher Pose Analysis', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return {
        'elbow_angle_correct': correct_elbow_angle_detected,
        'knee_angle_correct': correct_knee_angle_detected,
        'foot_direction_correct': correct_foot_direction_detected,
        'hip_rotation_correct': hip_rotates_faster,
        'movement_duration': timing_data['movement_duration'],
        'movement_too_slow': timing_data['movement_too_slow']
    }

def main():
    """主程式入口點"""
    #video_path = "IMG_7385.MOV"
    video_path = r"../baseball/src/Component/video/video.mp4"  # 請替換為您的影片路徑
    results = analyze_pitcher_pose(video_path)
    
    print("\n投球姿勢分析結果：")
    if results['elbow_angle_correct']:
        print("- 在手肘檢測階段，右手肘角度有達到正確範圍（160-170度）。")
    else:
        print("- 在手肘檢測階段，右手肘角度未達到正確範圍（160-170度）。")

    if results['knee_angle_correct']:
        print("- 在膝蓋檢測階段，左膝蓋接近伸直。")
    else:
        print("- 在膝蓋檢測階段，左膝蓋未充分伸直。")

    if results['foot_direction_correct']:
        print("- 在膝蓋檢測階段，腳尖有正確朝向捕手方向。")
    else:
        print("- 在膝蓋檢測階段，腳尖未正確朝向捕手方向。")

    if results['hip_rotation_correct']:
        print("- 在檢測過程中，髖部的旋轉速度有快於肩膀 => 達成肩髖分離。")
    else:
        print("- 在檢測過程中，髖部的旋轉速度未快於肩膀 => 未達成肩髖分離。")

    if results['movement_duration'] is not None:
        #print(f"- 投球動作時間：{results['movement_duration']:.2f}秒")
        if results['movement_too_slow']:
            print("  警告：動作時間過長！")

if __name__ == "__main__":
    main()