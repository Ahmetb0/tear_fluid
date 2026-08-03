import cv2
import numpy as np
import math
import csv
video_yolu = "C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv"
class BasitTakipci:
    def __init__(self):
        self.merkezler = {}
        self.id_sayaci = 0

    def guncelle(self, yeni_noktalar, fps):
        guncellenmis_merkezler = {}
        hizlar = {}
        
        # Her bir kare arasında geçen gerçek süreyi hesapla (Örn: 30 FPS için ~0.033 saniye)
        delta_t = 1.0 / fps 
        
        if len(self.merkezler) == 0:
            for nkt in yeni_noktalar:
                self.merkezler[self.id_sayaci] = nkt
                self.id_sayaci += 1
            return self.merkezler, hizlar

        for yeni_nkt in yeni_noktalar:
            min_mesafe = float('inf')
            eslesen_id = None
            
            for nesne_id, eski_nkt in self.merkezler.items():
                mesafe = math.hypot(yeni_nkt[0] - eski_nkt[0], yeni_nkt[1] - eski_nkt[1])
                if mesafe < min_mesafe and mesafe < 0.2: 
                    min_mesafe = mesafe
                    eslesen_id = nesne_id
            
            if eslesen_id is not None:
                guncellenmis_merkezler[eslesen_id] = yeni_nkt
                
                # Makaledeki standarda uygun MMS Hızı: (Mesafe / Kare_Süresi) * 0.1
                # Bu formül bize tam olarak 0.1 saniyedeki yer değiştirmeyi verir.
                mms = (min_mesafe / delta_t) * 0.1
                hizlar[eslesen_id] = mms
                
                del self.merkezler[eslesen_id]
            else:
                guncellenmis_merkezler[self.id_sayaci] = yeni_nkt
                self.id_sayaci += 1

        self.merkezler = guncellenmis_merkezler
        return self.merkezler, hizlar

def nihai_analiz_ve_takip(video_yolu):
    cap = cv2.VideoCapture(video_yolu)
    
    # 1. YENİLİK: Videonun gerçek hızını (FPS) al
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30.0 # Okuyamazsa varsayılan olarak 30 kabul et
        
    print(f"Video Hızı: {fps} FPS")
    print("İpucu: Çıkmak için 'q' tuşuna basabilirsiniz.")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    takipci = BasitTakipci()
    rotalar = {}

    while True:
        ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret:
            print("Video tamamlandı. Analiz bitiriliyor...")
            break

        orijinal_kare = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # =========================================================
        # REFERANS IŞIKLARI VE GÖZ KIRPMA (BLINK) TESPİTİ
        # =========================================================
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, ref_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        ref_contours, _ = cv2.findContours(ref_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ref_contours = sorted(ref_contours, key=cv2.contourArea, reverse=True)[:2]
        
        ust_isik = None
        d_cn = None
        goz_acik_mi = False # Göz kapalı varsayımıyla başlıyoruz

        if len(ref_contours) == 2:
            merkezler = []
            for rc in ref_contours:
                M_ref = cv2.moments(rc)
                if M_ref["m00"] != 0:
                    merkezler.append((int(M_ref["m10"] / M_ref["m00"]), int(M_ref["m01"] / M_ref["m00"])))
                    
            if len(merkezler) == 2:
                merkezler.sort(key=lambda nokta: nokta[1]) 
                ust_isik = merkezler[0] 
                alt_isik = merkezler[1] 
                d_cn = math.hypot(ust_isik[0] - alt_isik[0], ust_isik[1] - alt_isik[1])
                x_farki = abs(ust_isik[0] - alt_isik[0])
                y_farki = abs(ust_isik[1] - alt_isik[1])
                
                # 2. YENİLİK: İki ışık arasındaki mesafe belirli bir pikselin üzerindeyse göz açıktır
                if d_cn > 30 and y_farki > (x_farki * 2): 
                    goz_acik_mi = True
                    cv2.circle(orijinal_kare, ust_isik, 15, (255, 0, 0), 2)
                    cv2.circle(orijinal_kare, alt_isik, 15, (255, 0, 0), 2)
                    cv2.line(orijinal_kare, ust_isik, alt_isik, (255, 0, 0), 1)

        # GÖZ KAPANDIYSA VEYA KIRPILDIYSA: Tüm hafızayı sil, gürültüyü reddet
        if not goz_acik_mi:
            rotalar.clear()
            takipci = BasitTakipci()
            cv2.putText(orijinal_kare, "GOZ KIRPMA TESPITI - BEKLENIYOR...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Nihai TF Analizi", orijinal_kare)
            
            bekleme_suresi = max(1, int(1000 / fps))
            if cv2.waitKey(bekleme_suresi) & 0xFF == ord('q'):
                break
            continue # Göz açılana kadar aşağıdaki kodları (parçacık aramasını) atla

        # =========================================================
        # PARÇACIK BULMA VE NORMALİZASYON (Sadece göz tam açıkken çalışır)
        # =========================================================
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, particle_thresh = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
        p_contours, _ = cv2.findContours(particle_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        anlik_noktalar = [] 
        for cnt in p_contours:
            alan = cv2.contourArea(cnt)
            if 1 < alan < 50:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    X_n = (cX - ust_isik[0]) / d_cn
                    Y_n = (cY - ust_isik[1]) / d_cn
                    anlik_noktalar.append((X_n, Y_n, cX, cY))

        normalize_noktalar = [(n[0], n[1]) for n in anlik_noktalar]
        takip_edilen_nesneler, hizlar = takipci.guncelle(normalize_noktalar, fps)

        # =========================================================
        # ÇİZİM VE HIZ (MMS) GÖSTERİMİ
        # =========================================================
        for nesne_id, norm_nkt in takip_edilen_nesneler.items():
            gercek_cX, gercek_cY = 0, 0
            for n in anlik_noktalar:
                if n[0] == norm_nkt[0] and n[1] == norm_nkt[1]:
                    gercek_cX, gercek_cY = n[2], n[3]
                    break
            
            if gercek_cX == 0: continue 
            
            if nesne_id not in rotalar:
                rotalar[nesne_id] = []
            rotalar[nesne_id].append((gercek_cX, gercek_cY))
            
            for i in range(1, len(rotalar[nesne_id])):
                cv2.line(orijinal_kare, rotalar[nesne_id][i-1], rotalar[nesne_id][i], (0, 0, 255), 2)
            
            cv2.circle(orijinal_kare, (gercek_cX, gercek_cY), 4, (0, 255, 0), -1)
            
            mms_degeri = hizlar.get(nesne_id, 0.0)
            if mms_degeri > 0:
                yazi = f"ID:{nesne_id} V:{mms_degeri:.2f}"
                cv2.putText(orijinal_kare, yazi, (gercek_cX + 8, gercek_cY - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imshow("Nihai TF Analizi", orijinal_kare)
        
        # Gerçek zamanlı oynatmak için formül (Örn: 30 FPS ise her kare arası 33ms bekler)
        bekleme_suresi = max(1, int(1000 / fps))
        if cv2.waitKey(bekleme_suresi) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def nihai_analiz_ve_kayit(video_yolu, cikis_csv_adi="gozyasi_analizi.csv"):
    cap = cv2.VideoCapture(video_yolu)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30.0 
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    takipci = BasitTakipci()
    rotalar = {}
    
    # Videodaki zamanı takip etmek için kare sayacı
    kare_sayaci = 0 

    # 2. YENİLİK: CSV dosyasını yazma modunda (w) açıyoruz
    with open(cikis_csv_adi, mode='w', newline='') as dosya:
        yazici = csv.writer(dosya)
        # Tablonun başlık (Header) satırını yazdırıyoruz
        yazici.writerow(["Kare_No", "Zaman(sn)", "Parcacik_ID", "X_Norm", "Y_Norm", "MMS_Hizi"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break # Veri kaydederken başa sarmak yerine video bitince işlemi bitiriyoruz
            
            kare_sayaci += 1
            zaman_sn = kare_sayaci / fps # O anki süreyi saniye cinsinden hesapla

            orijinal_kare = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- REFERANS BULMA VE GÖZ KIRPMA KONTROLÜ (Önceki kodun aynısı) ---
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, ref_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
            ref_contours, _ = cv2.findContours(ref_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ref_contours = sorted(ref_contours, key=cv2.contourArea, reverse=True)[:2]
            
            ust_isik = None
            d_cn = None
            goz_acik_mi = False

            if len(ref_contours) == 2:
                merkezler = []
                for rc in ref_contours:
                    M_ref = cv2.moments(rc)
                    if M_ref["m00"] != 0:
                        merkezler.append((int(M_ref["m10"] / M_ref["m00"]), int(M_ref["m01"] / M_ref["m00"])))
                        
                if len(merkezler) == 2:
                    merkezler.sort(key=lambda nokta: nokta[1]) 
                    ust_isik = merkezler[0] 
                    alt_isik = merkezler[1] 
                    
                    d_cn = math.hypot(ust_isik[0] - alt_isik[0], ust_isik[1] - alt_isik[1])
                    x_farki = abs(ust_isik[0] - alt_isik[0])
                    y_farki = abs(ust_isik[1] - alt_isik[1])
                    
                    if d_cn > 30 and y_farki > (x_farki * 2): 
                        goz_acik_mi = True
                        cv2.circle(orijinal_kare, ust_isik, 15, (255, 0, 0), 2)
                        cv2.circle(orijinal_kare, alt_isik, 15, (255, 0, 0), 2)
                        cv2.line(orijinal_kare, ust_isik, alt_isik, (255, 0, 0), 1)

            if not goz_acik_mi:
                rotalar.clear()
                takipci = BasitTakipci()
                cv2.putText(orijinal_kare, "GOZ KIRPMA - VERI KAYDEDILMIYOR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Nihai TF Analizi", orijinal_kare)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue 

            # --- PARÇACIK BULMA VE TAKİP (Önceki kodun aynısı) ---
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            _, particle_thresh = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
            p_contours, _ = cv2.findContours(particle_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            anlik_noktalar = [] 
            for cnt in p_contours:
                alan = cv2.contourArea(cnt)
                if 1 < alan < 50:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        X_n = (cX - ust_isik[0]) / d_cn
                        Y_n = (cY - ust_isik[1]) / d_cn
                        anlik_noktalar.append((X_n, Y_n, cX, cY))

            normalize_noktalar = [(n[0], n[1]) for n in anlik_noktalar]
            takip_edilen_nesneler, hizlar = takipci.guncelle(normalize_noktalar, fps)

            # 3. YENİLİK: Çizim yaparken aynı zamanda verileri CSV'ye satır satır yazdırıyoruz
            for nesne_id, norm_nkt in takip_edilen_nesneler.items():
                mms_degeri = hizlar.get(nesne_id, 0.0)
                
                # Sadece hareket eden ve hızı hesaplanabilmiş parçacıkları kaydet
                if mms_degeri > 0:
                    yazici.writerow([kare_sayaci, round(zaman_sn, 3), nesne_id, round(norm_nkt[0], 4), round(norm_nkt[1], 4), round(mms_degeri, 4)])

                # (Çizim kodları buraya gelecek - orijinal_kare üzerine circle, line, putText vs.)
                # ...

            cv2.imshow("Nihai TF Analizi", orijinal_kare)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Analiz tamamlandı. Veriler '{cikis_csv_adi}' dosyasına kaydedildi.")

nihai_analiz_ve_takip(video_yolu)
nihai_analiz_ve_kayit(video_yolu)