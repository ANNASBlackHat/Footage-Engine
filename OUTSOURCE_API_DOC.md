Here is the verified API reference for **Pixabay**, **Pexels**, and **Coverr** based on their official API documentation.

---

### Quick Comparison Matrix

| Provider | Media Types | Orientation Filter Param | Date Filtering Support | Auth Method |
| --- | --- | --- | --- | --- |
| **Pixabay** | Photos, Illustrations, Vectors, Videos | `orientation=all|horizontal|vertical` | `order=latest` (sort only, no range 
filter) | Query param (`key=...`) |
| **Pexels** | Photos, Videos | `orientation=landscape|portrait|square` | ❌ No date filter parameter | Header (`Authorization: 
<KEY>`) |
| **Coverr** | Videos, Music, AI media | ❌ No filter param (returned in response: `is_vertical`, `aspect_ratio`) | `sort=date` (sort 
only, no range filter) | Header (`Authorization: Bearer <KEY>`) or Query param (`api_key=...`) |

---

### 1. Pixabay API

Official Docs: `[https://pixabay.com/api/docs/](https://pixabay.com/api/docs/)`

#### Endpoints

* **Images:** `GET [https://pixabay.com/api/](https://pixabay.com/api/)`
* **Videos:** `GET [https://pixabay.com/api/videos/](https://pixabay.com/api/videos/)`

#### Query Parameters

| Parameter | Type | Allowed Values / Details |
| --- | --- | --- |
| `key` *(required)* | string | Your API key. |
| `q` | string | URL-encoded search query (up to 100 characters). |
| `lang` | string | Language code (`en`, `es`, `de`, `fr`, `ja`, etc. Default: `en`). |
| `image_type` *(images only)* | string | `all` (default), `photo`, `illustration`, `vector`. |
| `video_type` *(videos only)* | string | `all` (default), `film`, `animation`. |
| `orientation` | string | `all` (default), `horizontal`, `vertical`. |
| `category` | string | `nature`, `science`, `people`, `places`, `animals`, `industry`, `food`, `sports`, `transportation`, `travel`, 
`buildings`, `business`, `music`, `backgrounds`, `fashion`, etc. |
| `min_width` / `min_height` | integer | Minimum width/height in pixels (Default: `0`). |
| `colors` *(images only)* | string | Comma-separated list: `grayscale`, `transparent`, `red`, `orange`, `yellow`, `green`, 
`turquoise`, `blue`, `lilac`, `pink`, `white`, `gray`, `black`, `brown`. |
| `editors_choice` | boolean | `true` or `false` (Default: `false`). |
| `safesearch` | boolean | `true` or `false` (Default: `false`). |
| `order` | string | `popular` (default) or `latest` (orders by newest upload date). *Specific date ranges are not supported.* |
| `page` | integer | Page number (Default: `1`). |
| `per_page` | integer | Results per page: `3` to `200` (Default: `20`). |

#### Example Request

```bash
# Search horizontal nature photos, sorted by latest
curl "https://pixabay.com/api/?key=YOUR_API_KEY&q=nature&image_type=photo&orientation=horizontal&order=latest&per_page=10"

```

---

### 2. Pexels API

Official Docs: `[https://www.pexels.com/api/documentation/](https://www.pexels.com/api/documentation/)`

#### Endpoints

* **Search Photos:** `GET [https://api.pexels.com/v1/search](https://api.pexels.com/v1/search)`
* **Curated Photos:** `GET [https://api.pexels.com/v1/curated](https://api.pexels.com/v1/curated)`
* **Search Videos:** `GET [https://api.pexels.com/videos/search](https://api.pexels.com/videos/search)`
* **Popular Videos:** `GET [https://api.pexels.com/videos/popular](https://api.pexels.com/videos/popular)`

#### Query Parameters

| Parameter | Applies To | Type | Allowed Values / Details |
| --- | --- | --- | --- |
| `query` *(required)* | Photos / Videos | string | Search keyword (e.g., `mountain`, `business`). |
| `orientation` | Photos / Videos | string | `landscape`, `portrait`, `square`. |
| `size` | Photos / Videos | string | `large` (24MP), `medium` (12MP), `small` (4MP). |
| `color` | Photos | string | Named color (`red`, `blue`, `black`, etc.) or Hex code (e.g., `#ffffff`). |
| `locale` | Photos / Videos | string | Search locale (e.g., `en-US`, `es-ES`, `de-DE`, `ja-JP`). |
| `min_width` / `min_height` | Videos | integer | Minimum width/height in pixels. |
| `min_duration` / `max_duration` | Videos | integer | Minimum/maximum duration in seconds. |
| `page` | Photos / Videos | integer | Page number (Default: `1`). |
| `per_page` | Photos / Videos | integer | Results per page (Default: `15`, Max: `80`). |

> **Date Filter Note:** Pexels does **not** offer a query parameter to filter or sort by specific upload dates.

#### Example Request

```bash
# Search vertical videos with duration between 5 and 30 seconds
curl -H "Authorization: YOUR_API_KEY" \
  "https://api.pexels.com/videos/search?query=workout&orientation=portrait&min_duration=5&max_duration=30&per_page=10"

```

---

### 3. Coverr API

Official Docs: `[https://api.coverr.co/docs/](https://api.coverr.co/docs/)`

#### Endpoints

* **List / Search Videos:** `GET [https://api.coverr.co/videos](https://api.coverr.co/videos)`
* **Get Video by ID:** `GET [https://api.coverr.co/videos/](https://api.coverr.co/videos/){id}`

#### Query Parameters

| Parameter | Type | Allowed Values / Details |
| --- | --- | --- |
| `query` | string | Text search query (e.g., `sunset`). |
| `sort` | string | `popular` (default), `date` (orders by newest upload date), `trending`. |
| `urls` | boolean | `true` or `false` (Default: `false`). Set to `true` to include direct signed download/preview URLs in list 
responses. |
| `page` | integer | Page number (**zero-based index**, Default: `0`). |
| `page_size` | integer | Number of items per page (Default: `20`). |
| `api_key` *(optional)* | string | API key query parameter (alternative to header auth). |

#### Orientation and Date Nuances on Coverr

* **Orientation:** Coverr does not accept an `orientation` parameter in the search endpoint. Instead, each returned item includes 
`"is_vertical": true|false` and `"aspect_ratio": "16:9"` in the response payload for client-side filtering.
* **Date Filtering:** You can sort by newest assets using `sort=date`, but there is no date range filter (such as `start_date` / 
`end_date`).

#### Example Request

```bash
# Search videos sorted by date with download URLs enabled
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "https://api.coverr.co/videos?query=city&sort=date&urls=true&page=0&page_size=10"

```
