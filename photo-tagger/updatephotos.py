from metaflow import (
    card,
    FlowSpec,
    step,
    current,
    project,
    Flow,
    retry,
    secrets,
    schedule,
    Parameter,
)
import os, time, json
from datetime import datetime, timezone
from dateutil.parser import isoparse
from metaflow.cards import Markdown, Image
import random

import requests

PHOTOS_URL = "https://api.unsplash.com/photos"
SEARCH_URL = "https://api.unsplash.com/search/photos"
COLLECTIONS_URL = "https://api.unsplash.com/collections"


def request(path="", search_query=None, page=1, per_page=30, collection_id=None):
    unsplash_client_id = os.environ["client_id"]
    
    params = {
        "client_id": unsplash_client_id,
        "page": page,
        "per_page": per_page
    }
    
    if search_query:
        params["query"] = search_query
    
    if search_query and not path.startswith("search"):
        url = SEARCH_URL
    elif collection_id:
        url = f"{COLLECTIONS_URL}/{collection_id}/photos"
    elif path == "random":
        url = "https://api.unsplash.com/photos/random"
        params["count"] = per_page
    else:
        url = os.path.join(PHOTOS_URL, path)
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        
        if search_query and "results" in result:
            return result["results"]
        elif path == "random":
            return result if isinstance(result, list) else [result]
        else:
            return result
    except Exception as e:
        print(f"API request failed: {e}")
        return []


def get_diverse_photos():
    hour = datetime.now().hour
    day = datetime.now().day
    minute = datetime.now().minute
    
    search_terms = [
        "nature", "architecture", "food", "travel", "portrait", 
        "landscape", "urban", "abstract", "technology", "art",
        "animals", "flowers", "mountains", "ocean", "city",
        "vintage", "minimal", "colorful", "black and white", "sunset"
    ]
    
    collections = [
        "3816141", "1319040", "1114848", "1065976", "162326",
        "1154337", "1154338", "827743", "1154339", "1154340"
    ]
    
    strategies = [
        {"path": "", "page": (hour % 10) + 1},
        {"path": "", "page": (day % 15) + 1},
        {"path": "random", "per_page": 30},
        {"search_query": search_terms[hour % len(search_terms)], "page": (minute % 3) + 1},
        {"search_query": search_terms[day % len(search_terms)], "page": 1},
        {"search_query": search_terms[(hour + day) % len(search_terms)], "page": 2},
        {"collection_id": collections[hour % len(collections)]},
        {"collection_id": collections[day % len(collections)]},
        {"collection_id": collections[(hour + minute) % len(collections)]},
    ]
    
    strategy_index = (hour * 60 + minute) % len(strategies)
    strategy = strategies[strategy_index]
    
    print(f"Using strategy {strategy_index + 1}: {strategy}")
    
    photos = request(**strategy)
    
    if not photos:
        print("Strategy failed, falling back to basic request")
        photos = request()
    
    return photos if photos else []


def photo_info(photo):
    try:
        photo_id = photo.get("id")
        slug = photo.get("slug", photo_id)
        
        if not photo_id:
            return None
            
        details = request(slug)
        if not details or not details.get("tags"):
            return None
            
        return photo_id, {
            "id": photo_id,
            "label": photo.get("alt_description") or photo.get("description", ""),
            "image_url": details["urls"]["regular"],
            "ground_truth_tags": [t["title"].lower() for t in details["tags"] if t.get("title")],
            "timestamp": details.get("updated_at", datetime.now().isoformat()),
            "attribution_url": details["links"]["html"],
            "attribution_text": details["user"]["name"],
        }
    except Exception as e:
        print(f"Error processing photo {photo.get('id', 'unknown')}: {e}")
        return None


def get_thumb(photo):
    return requests.get(photo["thumb"]).content


def prune_old(photos):
    now = datetime.now(timezone.utc)
    return {
        k: v for k, v in photos.items() if (now - isoparse(v["timestamp"])).days < 30
    }


def get_historical_counts(first, max_num=48):
    counts = [first] + [0] * (max_num - 1)
    i = 1
    for run in Flow("UpdatePhotos"):
        if "start" in run and "unseen" in run["start"].task:
            counts[i] = len(run["start"].task["unseen"].data)
            i += 1
            if i == max_num:
                break

    maxx = max(counts)
    lst = [{"perc": int(100 * c / maxx)} for c in counts]
    lst[0]["isFirst"] = True
    return lst


class SkipTrigger(Exception):
    pass


@schedule(hourly=True)
@project(name="photo_tagger")
class UpdatePhotos(FlowSpec):

    reset_existing = Parameter("reset-existing", default="no")
    diversity_mode = Parameter("diversity-mode", default="enhanced", 
                              help="Data collection mode: 'basic' or 'enhanced'")
    allow_empty = Parameter("allow-empty", default=False)

    @retry
    @secrets(sources=["outerbounds.unsplash"])
    @card(type="phototagger")
    @step
    def start(self):
        try:
            existing = Flow("UpdatePhotos").latest_successful_run.data.photos
        except:
            print("No existing photos found - starting from scratch")
            existing = {}
        if self.reset_existing != "no":
            print("Forgetting existing photos")
            existing = {}
        
        if self.diversity_mode == "enhanced":
            photos = get_diverse_photos()
        else:
            photos = request()
        
        new_photos = []
        for photo in photos:
            if photo and photo.get("id") not in existing:
                photo_data = photo_info(photo)
                if photo_data:
                    new_photos.append(photo_data)
        
        self.unseen = dict(filter(lambda x: x, new_photos))
        
        print(f"Found {len(photos)} photos from API")
        print(f"Processed {len(self.unseen)} new photos with tags")
        print(f"Total unique photos: {len(existing) + len(self.unseen)}")
        
        counts = get_historical_counts(len(self.unseen))
        self.photo_grid = {
            "num": len(counts),
            "counts": counts,
            "num_new": len(photos),
            "num_processed": len(self.unseen),
            "diversity_mode": self.diversity_mode,
            "updated_at": datetime.now().isoformat().split(".")[0],
            "photos": [{"image_url": p["image_url"]} for p in self.unseen.values()],
        }
        self.photos = self.unseen | prune_old(existing)
        
        if self.unseen:
            from assets import Asset

            Asset(project="photo_tagger", branch="main").register_data_asset(
                "photos", "Latest set of photos", "artifact", ["unseen"]
            )
            print("New asset instance registered")

        self.next(self.end)

    @step
    def end(self):
        if self.unseen:
            print("Finishing the run successfully to create an event")
        elif self.allow_empty:
            print("No new photos – exiting cleanly for CI")
        else:
            raise SkipTrigger(
                "Not an error - failing this run on purpose "
                "to avoid triggering flows downstream"
            )


if __name__ == "__main__":
    UpdatePhotos()
