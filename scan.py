import os
import re
import json
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import cv2
import requests
from googleapiclient.discovery import build

YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]

# ---------------------------------------------------------------------------
# AYARLANABİLİR PARAMETRELER — bunları değiştirerek taramayı sıkılaştırıp
# gevşetebilirsin.
# ---------------------------------------------------------------------------
REGIONS = ["US", "GB", "CA", "AU", "IE", "NZ"]          # İngilizce konuşulan bölgeler
CATEGORY_IDS = ["", "1", "2", "10", "15", "17", "19",     # "" = kategori filtresi yok
                "20", "22", "23", "24", "25", "26", "27", "28"]
MAX_SUBSCRIBERS = 20_000         # "az abone" eşiği — bunun üstü elenir
IDEAL_SUBSCRIBERS = 10_000       # Bu ve altı "🌟 İDEAL" olarak işaretlenir
MAX_CHANNEL_VIDEO_COUNT = 30     # Kanalın toplam video sayısı bu sayıdan az olmalı
MIN_VIEW_TO_SUB_RATIO = 10       # İzlenme, abone sayısının en az kaç katı olmalı
MIN_VIDEO_SECONDS = 60           # Shorts'u elemek için (yatay/normal video şartı)
MAX_VIDEO_AGE_DAYS = 21          # "yakın zamanda" şartı
TOP_N_RESULTS = 30
EXCLUDE_FACES = True             # True: thumbnail'de insan yüzü görünen videolar elenir (faceless niş şartı)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def parse_duration(iso_duration: str) -> int:
    """ISO 8601 süresini saniyeye çevirir (örn: PT5M30S -> 330)."""
    pattern = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
    match = pattern.match(iso_duration or "")
    if not match:
        return 0
    hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def fetch_trending_videos():
    """Tüm bölge/kategori kombinasyonlarında 'mostPopular' listesini çeker."""
    seen_ids = set()
    videos = []
    for region in REGIONS:
        for category_id in CATEGORY_IDS:
            params = {
                "part": "snippet,statistics,contentDetails",
                "chart": "mostPopular",
                "regionCode": region,
                "maxResults": 50,
            }
            if category_id:
                params["videoCategoryId"] = category_id
            try:
                response = youtube.videos().list(**params).execute()
            except Exception:
                # Bazı kategoriler bazı bölgelerde geçersiz olabilir, atla
                continue

            for item in response.get("items", []):
                vid = item["id"]
                if vid in seen_ids:
                    continue
                seen_ids.add(vid)
                videos.append(item)
            time.sleep(0.1)
    return videos


def filter_videos(videos):
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_VIDEO_AGE_DAYS)
    filtered = []
    for v in videos:
        duration_sec = parse_duration(v["contentDetails"]["duration"])
        if duration_sec < MIN_VIDEO_SECONDS:
            continue  # Shorts'u ele (dikey/kısa videolar)
        try:
            published_at = datetime.strptime(
                v["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if published_at < cutoff:
            continue  # eski videoları ele
        filtered.append(v)
    return filtered


def fetch_channel_stats(channel_ids):
    stats_by_channel = {}
    channel_ids = list(set(channel_ids))
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i + 50]
        try:
            response = youtube.channels().list(
                part="statistics,snippet", id=",".join(batch)
            ).execute()
        except Exception:
            continue
        for item in response.get("items", []):
            stats = item.get("statistics", {})
            hidden = stats.get("hiddenSubscriberCount", False)
            stats_by_channel[item["id"]] = {
                "subscriberCount": None if hidden else int(stats.get("subscriberCount", 0)),
                "videoCount": int(stats.get("videoCount", 0)),
                "channelTitle": item["snippet"]["title"],
            }
    return stats_by_channel


def thumbnail_has_face(thumbnail_url: str) -> bool:
    """Video kapağında (thumbnail) GERÇEK bir insan yüzü var mı kontrol eder.
    Amaç: AI avatar, çizgi/illüstrasyon, 3D render gibi karakterleri ELEMEMEK,
    sadece gerçek insan fotoğrafı/yüzü gösteren videoları elemek (faceless niş şartı).
    Haar Cascade dedektörü gerçek fotoğraflardaki ışık/gölge/doku desenlerine göre
    eğitildiği için stilize/çizgi/AI-render karakterlerde genelde tetiklenmez —
    bu da tam olarak istediğimiz davranış. %100 kusursuz değildir; çok fotogerçekçi
    AI yüzler nadiren yanlışlıkla yakalanabilir, çizgi karakterler nadiren yanlış
    pozitif verebilir."""
    try:
        response = requests.get(
            thumbnail_url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        response.raise_for_status()
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return True  # görsel okunamadıysa güvenli tarafta kal, hariç tut
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return len(faces) > 0
    except Exception:
        return True  # kontrol başarısız olduysa güvenli tarafta kal, hariç tut


def build_opportunities(videos, channel_data):
    opportunities = []
    for v in videos:
        channel_id = v["snippet"]["channelId"]
        info = channel_data.get(channel_id)
        if not info or info["subscriberCount"] is None:
            continue
        subs = info["subscriberCount"]
        video_count = info["videoCount"]
        if subs == 0 or subs > MAX_SUBSCRIBERS:
            continue
        if video_count > MAX_CHANNEL_VIDEO_COUNT:
            continue  # kanalda zaten çok video varsa "fırsat" değil
        views = int(v["statistics"].get("viewCount", 0))
        ratio = views / max(subs, 1)
        if ratio < MIN_VIEW_TO_SUB_RATIO:
            continue  # yeterince "patlamamış"
        if EXCLUDE_FACES:
            thumbs = v["snippet"].get("thumbnails", {})
            thumb_url = (
                thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
            ).get("url")
            if thumb_url and thumbnail_has_face(thumb_url):
                continue  # gerçek insan yüzü tespit edildi, faceless şartına uymuyor
        opportunities.append({
            "score": round(ratio, 2),
            "channel": info["channelTitle"],
            "subscribers": subs,
            "channel_video_count": video_count,
            "ideal": subs <= IDEAL_SUBSCRIBERS,
            "title": v["snippet"]["title"],
            "views": views,
            "published": v["snippet"]["publishedAt"][:10],
            "url": f"https://www.youtube.com/watch?v={v['id']}",
        })
    opportunities.sort(key=lambda x: x["score"], reverse=True)
    return opportunities[:TOP_N_RESULTS]


def main():
    videos = fetch_trending_videos()
    filtered = filter_videos(videos)
    channel_ids = [v["snippet"]["channelId"] for v in filtered]
    channel_data = fetch_channel_stats(channel_ids)
    opportunities = build_opportunities(filtered, channel_data)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria": {
            "max_subscribers": MAX_SUBSCRIBERS,
            "ideal_subscribers": IDEAL_SUBSCRIBERS,
            "max_channel_video_count": MAX_CHANNEL_VIDEO_COUNT,
            "min_view_to_sub_ratio": MIN_VIEW_TO_SUB_RATIO,
            "max_video_age_days": MAX_VIDEO_AGE_DAYS,
        },
        "opportunities": opportunities,
    }

    os.makedirs("docs", exist_ok=True)
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"{len(opportunities)} fırsat niş bulundu.")


if __name__ == "__main__":
    main()
