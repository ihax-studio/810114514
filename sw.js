// sw.js
const CACHE_NAME = 'homo-waipu-v1';
const FILES_TO_CACHE = [
  '/',
  '/index.html',
  '/yaju.css', // 必要なCSS
  '/main.js',  // 必要なら
  '/bg.png',
  '/b1.png','/b2.png','/b3.png','/b4.png','/b5.png','/b6.png','/b7.png','/b8.png','/b9.png','/b10.png','/b11.png',
  '/ball.png','/114514.png','/red.png',
  '/icon-192.png','/icon-512.png','/icon-191.png',
  '/cc.mp3','/wrong.mp3','/mac.wav','/mac.mp3','/key.mp3','/collect.mp3','/change.mp3',
  // 他の使うファイル全部！（音・画像・JSも）
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(FILES_TO_CACHE))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
    ))
  );
  self.clients.claim();
});

// キャッシュ優先
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request, {ignoreSearch:true})
      .then(response => response || fetch(event.request))
  );
});
