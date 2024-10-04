import sys
import numpy as np
import cv2

# 입력 영상 불러오기
src = cv2.imread('paka.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

# 사각형 지정을 통한 초기 분할
rc = cv2.selectROI(src)  # ROI Selector 창이 뜨면 사각형 영역을 지정해주면 됨
mask = np.zeros(src.shape[:2], np.uint8)  # 검정색으로 입력 영상과 동일한 크기의 mask 생성

cv2.grabCut(src, mask, rc, None, None, 5, cv2.GC_INIT_WITH_RECT)
# 5번 iteration

# 0: cv2.GC_BGD, 2: cv2.GC_PR_BGD
mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
# 0과 2가 background이므로 background는 0으로, foreground는 1로 set
dst = src * mask2[:, :, np.newaxis] + (1 - mask2)[:, :, np.newaxis] * 255 #dst를 더 선명하게 보기 위해 배경 흰색으로 변경

# 초기 분할 결과 출력
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
