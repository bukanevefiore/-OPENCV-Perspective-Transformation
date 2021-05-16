
# VİDEO Link:  https://www.youtube.com/watch?v=6rPYnlZaYA0

# Papatya resminizi opencv kütüphanesinin perspective transformation yöntemiyle reklam paneline yerleştirmek 
# için fare ile panel resmi üzerinden aldığımız koordinatlara göre dönüşümü gerçekleştiricez
# öncelikle gerekli kütüphaneleri import ediyoruz
import cv2
import numpy as np

# Panonun yerleştirilmesi gereken koordinatları depolamak için değişkenleri tanımlama
positions=[] 
positions2=[]
count=0

# Bir fare tıklaması olayı kullanarak panodan dört koordinatı alma
# Fare geri çağırma işlevi
def draw_circle(event,x,y,flags,param):
    global positions,count
    # Olay Sol Düğmeyse Tıklayın ve koordinatı listelerde saklayın
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(building,(x,y),2,(255,0,0),-1)
        positions.append([x,y])
        if(count!=3):
            positions2.append([x,y])
        elif(count==3):
            positions2.insert(2,[x,y])
        count+=1
        
# İki görüntünün okunması ve değişkende (building ve dp'de) depolanması
building = cv2.imread('pano.jpg')
dp = cv2.imread('pa.jpg')

# 'Görüntü' adlı bir pencere tanımlama
cv2.namedWindow('image')

cv2.setMouseCallback('image',draw_circle)

while(True):
    cv2.imshow('image',building)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

#  Papatyanın genişliğini ve yüksekliğini w1 ve h1 de saklama
height, width = building.shape[:2]
h1,w1 = dp.shape[:2]

pts1=np.float32([[0,0],[w1,0],[0,h1],[w1,h1]])
pts2=np.float32(positions)

# Artık resimlerin koordinatlarına sahibiz. Cv2.findHomography () işlevini kullanarak 
# homografi matrisini hesaplayacağız.
h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)

height, width, channels = building.shape

# Şimdi image1 (dp), homografi matrisi (h) ve ikinci görüntünün genişlik ve yükseklik 
# girdilerini alacak olan cv2.warpPerspective () işlevini kullanacağız ve aşağıdaki çıktıyı vereceğiz.
im1Reg = cv2.warpPerspective(dp, h, (width, height))
cv2.imwrite('mask1.png',im1Reg)

# Pano görüntüsünün boyutuna göre np.zeros () kullanarak bir maske yapacağız. 
# Bu, aşağıdaki şekilde gösterildiği gibi olacaktır.
mask2 = np.zeros(building.shape, dtype=np.uint8)
cv2.imwrite('mask2.png',mask2)

roi_corners2 = np.int32(positions2)
channel_count2 = building.shape[2]  
ignore_mask_color2 = (255,)*channel_count2

# Şimdi cv2.fillConvexPoly () işlevini kullanarak maskeyi beyaz alanla dolduracağız 
# ve çıktıyı gösterildiği gibi döndüreceğiz.
cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)
cv2.imwrite('mask3.png',mask2)

# Şimdi elde edilen maskeyi cv2.bitwise_not () fonksiyonunu kullanarak tersine çevireceğiz.
mask2 = cv2.bitwise_not(mask2)
cv2.imwrite('mask4.png',mask2)

masked_image2 = cv2.bitwise_and(building, mask2)
cv2.imwrite('mask5.png',masked_image2)

# Son çıktıyı elde etmek için cv2.bitwise_or () işlevini kullanacağız.
final = cv2.bitwise_or(im1Reg, masked_image2)
cv2.imshow("Final", final)
cv2.imwrite('mask6.png',final)

cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

