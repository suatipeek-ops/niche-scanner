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
# AYARLANABİLİR PARAMETRELER
# ---------------------------------------------------------------------------
REGION_CODE = "US"
MIN_SUBSCRIBERS = 500              # Bunun altı: henüz çok yeni/test aşaması, güvenilir sinyal yok
MAX_SUBSCRIBERS = 20_000           # Bunun üstü: "az abone" şartına uymuyor
MAX_CHANNEL_VIDEO_COUNT = 20       # Kanalın toplam video sayısı bu sayıdan fazla olamaz
MIN_VIDEO_SECONDS = 600            # 10 dakika — bu süre Shorts'u zaten kesin olarak eler
MAX_VIDEO_SECONDS = 1800           # 30 dakika
MIN_VIEW_TO_SUB_RATIO = 40         # İzlenme, abone sayısının en az kaç katı olmalı (40-50x)
SEARCH_LOOKBACK_DAYS = 730         # Arama için makul bir tarih penceresi (2 yıl)
TOP_N_RESULTS = 30
EXCLUDE_FACES = True               # Gerçek insan yüzü tespit edilirse kanal tamamen elenir
EXCLUDED_VIDEO_CATEGORY_IDS = {"10"}             # Music — niş olarak asla kabul edilmez
EXCLUDED_CHANNEL_KEYWORDS = ["- topic", "vevo"]  # Otomatik müzik dağıtım / resmi label kanalları
REQUIRE_ENGLISH_IF_KNOWN = True

# Geniş, müzik dışı niş/konu havuzu — her gün hepsi taranır.
SEARCH_TOPICS = [
    "forgotten history facts", "ancient civilizations mystery", "unsolved disappearances case",
    "true crime cold case", "FBI declassified files", "ghost stories real encounters",
    "abandoned places explored", "urban legends explained", "conspiracy theories explained",
    "dark psychology facts", "body language secrets", "stoicism life lessons",
    "philosophy explained simply", "quantum physics explained", "black holes explained",
    "space exploration facts", "ocean mysteries unexplained", "deep sea creatures facts",
    "animal facts you didn't know", "wildlife survival stories", "world war 2 stories",
    "cold war secrets", "declassified cia operations", "lost civilizations history",
    "mythology stories explained", "ancient rome facts", "ancient egypt secrets",
    "medieval history facts", "geography facts surprising", "economics explained simply",
    "investing for beginners", "personal finance tips", "side hustle ideas",
    "real estate investing basics", "cryptocurrency explained simply", "business case study failure",
    "billionaire habits success", "minimalism lifestyle tips", "productivity hacks science",
    "self improvement habits", "psychology facts about people", "true crime serial killer case",
    "missing persons mystery solved", "weird laws around the world", "scary true stories reddit",
    "life hack tips daily", "science facts mind blowing", "history documents declassified",
    "famous heists explained", "court case true story",
]

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def parse_duration(iso_duration: str) -> int:
    """ISO 8601 süresini saniyeye çevirir (örn: PT5M30S -> 330)."""
    pattern = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
    match = pattern.match(iso_duration or "")
    if not match:
        return 0
    hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def thumbnail_has_face(thumbnail_url: str) -> bool:
    """Video kapağında GERÇEK bir insan yüzü var mı kontrol eder.
    AI avatar / çizgi / 3D render karakterleri ELEMEZ, sadece gerçek insan
    fotoğrafı gösteren videoları eler (faceless niş şartı). İndirme/okuma
    başarısız olursa güvenli tarafta kalınır (yüz var kabul edilir)."""
    try:
        response = requests.get(thumbnail_url, timeout=10, headers=HTTP_HEADERS)
        response.raise_for_status()
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return len(faces) > 0
    except Exception:
        return True


def is_valid_video(v) -> bool:
    """Bir videonun temel kalite şartlarını kontrol eder:
    müzik değil, dili biliniyorsa İngilizce, süre 10-30 dakika aralığında
    (bu süre aralığı Shorts'u zaten kesin olarak eler, ekstra kontrol gerekmez)."""
    snippet = v["snippet"]
    if snippet.get("categoryId") in EXCLUDED_VIDEO_CATEGORY_IDS:
        return False
    if REQUIRE_ENGLISH_IF_KNOWN:
        lang = snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage")
        if lang and not lang.lower().startswith("en"):
            return False
    duration_sec = parse_duration(v["contentDetails"]["duration"])
    if not (MIN_VIDEO_SECONDS <= duration_sec <= MAX_VIDEO_SECONDS):
        return False
    return True


def parse_iso8601(timestamp: str) -> datetime:
    """YouTube API'den gelen ISO 8601 tarihlerini ayrıştırır.
    Bazı tarihler ondalık saniye içerir (...29.371972Z), bazıları içermez —
    ikisini de doğru şekilde okur."""
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def chunked(items, size=50):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def search_candidate_video_ids():
    """Geniş konu havuzuyla arama yapıp benzersiz video ID'leri toplar.
    Konuların yarısı 'relevance' (ilgi/otorite), yarısı 'date' (en yeni
    yüklenenler) sıralamasıyla aranır — sadece relevance kullanmak zaten
    köklü/büyük kanalları öne çıkarma eğiliminde, bu da henüz arama
    otoritesi kazanmamış yeni/küçük kanalları kaçırmamıza sebep oluyordu."""
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=SEARCH_LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    video_ids = set()
    for i, topic in enumerate(SEARCH_TOPICS):
        order = "relevance" if i % 2 == 0 else "date"
        try:
            response = youtube.search().list(
                part="snippet",
                q=topic,
                type="video",
                order=order,
                publishedAfter=cutoff_iso,
                relevanceLanguage="en",
                regionCode=REGION_CODE,
                maxResults=50,
            ).execute()
        except Exception:
            continue
        for item in response.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid:
                video_ids.add(vid)
        time.sleep(0.1)
    return list(video_ids)


def fetch_videos_details(video_ids):
    """Verilen video ID'leri için tam detayları (snippet, statistics, duration) çeker."""
    videos = []
    for batch in chunked(video_ids, 50):
        try:
            response = youtube.videos().list(
                part="snippet,statistics,contentDetails", id=",".join(batch)
            ).execute()
        except Exception:
            continue
        videos.extend(response.get("items", []))
    return videos


def fetch_channel_info(channel_ids):
    """Kanal abone sayısı, video sayısı, açılış tarihi ve uploads playlist ID'sini çeker."""
    info_by_channel = {}
    for batch in chunked(list(set(channel_ids)), 50):
        try:
            response = youtube.channels().list(
                part="snippet,statistics,contentDetails", id=",".join(batch)
            ).execute()
        except Exception:
            continue
        for item in response.get("items", []):
            stats = item.get("statistics", {})
            hidden = stats.get("hiddenSubscriberCount", False)
            uploads_playlist = (
                item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
            )
            info_by_channel[item["id"]] = {
                "subscriberCount": None if hidden else int(stats.get("subscriberCount", 0)),
                "videoCount": int(stats.get("videoCount", 0)),
                "channelTitle": item["snippet"]["title"],
                "channelCreated": item["snippet"]["publishedAt"],
                "uploadsPlaylist": uploads_playlist,
            }
    return info_by_channel


def fetch_channel_uploads(uploads_playlist_id):
    """Bir kanalın yüklediği videoların ID + yayın tarihini çeker (max 20 video şartı
    sayesinde tek sayfada tamamı gelir)."""
    try:
        response = youtube.playlistItems().list(
            part="contentDetails", playlistId=uploads_playlist_id, maxResults=50
        ).execute()
    except Exception:
        return []
    items = []
    for item in response.get("items", []):
        cd = item.get("contentDetails", {})
        vid = cd.get("videoId")
        published = cd.get("videoPublishedAt")
        if vid and published:
            items.append((vid, published))
    return items


def evaluate_channel(channel_id, info):
    """Bir kanalın tüm 'faceless niş' şartlarını sağlayıp sağlamadığını kontrol eder."""
    subs = info["subscriberCount"]
    if subs is None or not (MIN_SUBSCRIBERS <= subs <= MAX_SUBSCRIBERS):
        return None
    if info["videoCount"] > MAX_CHANNEL_VIDEO_COUNT:
        return None
    channel_title_lower = info["channelTitle"].lower()
    if any(kw in channel_title_lower for kw in EXCLUDED_CHANNEL_KEYWORDS):
        return None
    if not info["uploadsPlaylist"]:
        return None

    uploads = fetch_channel_uploads(info["uploadsPlaylist"])
    if not uploads:
        return None

    upload_video_ids = [vid for vid, _ in uploads]
    upload_dates = [parse_iso8601(pub) for _, pub in uploads]
    earliest_video_date = min(upload_dates)
    channel_created = parse_iso8601(info["channelCreated"])
    now = datetime.now(timezone.utc)
    channel_age_days = (now - channel_created).days
    first_video_age_days = (now - earliest_video_date).days

    video_details = fetch_videos_details(upload_video_ids)

    valid_videos = []
    for v in video_details:
        if not is_valid_video(v):
            continue
        thumbs = v["snippet"].get("thumbnails", {})
        thumb_url = (
            thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}
        ).get("url")
        views = int(v["statistics"].get("viewCount", 0))
        valid_videos.append({
            "title": v["snippet"]["title"],
            "views": views,
            "ratio": round(views / max(subs, 1), 1),
            "published": v["snippet"]["publishedAt"][:10],
            "url": f"https://www.youtube.com/watch?v={v['id']}",
            "thumbnail": thumb_url,
        })

    if not valid_videos:
        return None

    qualifying = [vv for vv in valid_videos if vv["ratio"] >= MIN_VIEW_TO_SUB_RATIO]
    if not qualifying:
        return None  # hiçbir video abone sayısının yeterince katı izlenmemiş

    if EXCLUDE_FACES:
        for vv in valid_videos:
            if vv["thumbnail"] and thumbnail_has_face(vv["thumbnail"]):
                return None  # gerçek yüz tespit edildi, kanal tamamen elenir

    for vv in valid_videos:
        vv.pop("thumbnail", None)

    valid_videos.sort(key=lambda x: x["ratio"], reverse=True)
    best_ratio = max(vv["ratio"] for vv in qualifying)
    high_ratio_count = len(qualifying)

    return {
        "channel": info["channelTitle"],
        "subscribers": subs,
        "channel_video_count": info["videoCount"],
        "channel_age_days": channel_age_days,
        "first_video_age_days": first_video_age_days,
        "best_ratio": best_ratio,
        "high_ratio_count": high_ratio_count,
        "top_videos": valid_videos[:10],
    }


def main():
    print("Konu havuzunda arama yapılıyor...")
    candidate_video_ids = search_candidate_video_ids()
    print(f"{len(candidate_video_ids)} benzersiz video bulundu, detaylar çekiliyor...")

    candidate_videos = fetch_videos_details(candidate_video_ids)
    candidate_videos = [v for v in candidate_videos if is_valid_video(v)]

    channel_ids = list({v["snippet"]["channelId"] for v in candidate_videos})
    print(f"{len(channel_ids)} benzersiz kanal bulundu, kanal bilgileri çekiliyor...")
    channel_info = fetch_channel_info(channel_ids)

    pre_filtered = {
        cid: info for cid, info in channel_info.items()
        if info["subscriberCount"] is not None
        and MIN_SUBSCRIBERS <= info["subscriberCount"] <= MAX_SUBSCRIBERS
        and info["videoCount"] <= MAX_CHANNEL_VIDEO_COUNT
        and not any(kw in info["channelTitle"].lower() for kw in EXCLUDED_CHANNEL_KEYWORDS)
    }
    print(f"{len(pre_filtered)} kanal ön filtreyi geçti, derinlemesine inceleniyor...")

    opportunities = []
    for cid, info in pre_filtered.items():
        result = evaluate_channel(cid, info)
        if result:
            opportunities.append(result)

    opportunities.sort(key=lambda x: (x["high_ratio_count"], x["best_ratio"]), reverse=True)
    opportunities = opportunities[:TOP_N_RESULTS]

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria": {
            "min_subscribers": MIN_SUBSCRIBERS,
            "max_subscribers": MAX_SUBSCRIBERS,
            "max_channel_video_count": MAX_CHANNEL_VIDEO_COUNT,
            "min_video_minutes": MIN_VIDEO_SECONDS // 60,
            "max_video_minutes": MAX_VIDEO_SECONDS // 60,
            "min_view_to_sub_ratio": MIN_VIEW_TO_SUB_RATIO,
        },
        "opportunities": opportunities,
    }

    os.makedirs("docs", exist_ok=True)
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"{len(opportunities)} fırsat niş kanalı bulundu.")


if __name__ == "__main__":
    main()
