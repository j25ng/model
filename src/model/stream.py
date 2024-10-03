import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 비디오 프레임을 처리하는 클래스 정의
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # 프레임을 numpy 배열로 변환
        self.frame = img  # 이미지 저장
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit 사용자 인터페이스
st.title("웹캠으로 이미지 캡처하기")

# WebRTC 스트리머 실행
ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# 사용자가 '캡처' 버튼을 클릭하면 이미지 저장
if ctx.video_transformer:
    if st.button("캡처"):
        captured_frame = ctx.video_transformer.frame
        if captured_frame is not None:
            st.image(captured_frame, channels="BGR")
        else:
            st.warning("웹캠에서 이미지를 가져오지 못했습니다.")
