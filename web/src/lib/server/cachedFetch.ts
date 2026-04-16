const CACHE_DURATION_MS = 15 * 60 * 1000;
const MAX_CACHE_ITEMS = 100;

type CacheItem = {
  path: string;
  timestamp: Date;
  response: Response;
};

const pathCache = new Map<string, CacheItem>();
const timeStampCache: CacheItem[] = [];

export async function cachedFetch(
  input: URL | RequestInfo,
  init?: RequestInit,
): Promise<Response> {
  const method =
    init?.method ?? (input instanceof Request ? input.method : "GET");

  // Only cache GET requests
  if (method.toUpperCase() !== "GET") {
    return fetch(input, init);
  }

  const path =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;

  // Process timestampCache
  let staleCount = 0;
  let currentTime = new Date();
  for (let i = 0; i < timeStampCache.length; i++) {
    // Items should be ordered, so break upon hitting first non-stale response
    if (
      currentTime.valueOf() - timeStampCache[i].timestamp.valueOf() <
      CACHE_DURATION_MS
    ) {
      break;
    }

    staleCount++;
  }

  // Delete stale responses from the pathCache
  for (let i = 0; i < staleCount; i++) {
    const staleItem = timeStampCache[i];
    let staleKey = staleItem.path;
    pathCache.delete(staleKey);
  }

  // Delete stale responses from the timestamp cache
  if (staleCount > 0) {
    timeStampCache.splice(0, staleCount);
  }

  let key = path;
  let savedData: CacheItem | undefined = pathCache.get(key);

  if (savedData !== undefined) {
    if (
      currentTime.valueOf() - savedData.timestamp.valueOf() <
      CACHE_DURATION_MS
    ) {
      return savedData.response.clone();
    }
  }

  // Data is either not cached or stale, so do a normal fetch
  let response = await fetch(input, init);

  let cacheItem: CacheItem = {
    path,
    timestamp: new Date(),
    response: response.clone(),
  };

  // Prevent memory leak
  if (timeStampCache.length >= MAX_CACHE_ITEMS) {
    const oldestItems = timeStampCache.splice(
      0,
      timeStampCache.length - MAX_CACHE_ITEMS,
    );

    for (const item of oldestItems) {
      pathCache.delete(item.path);
    }
  }

  pathCache.set(key, cacheItem);
  timeStampCache.push(cacheItem);

  return response;
}
