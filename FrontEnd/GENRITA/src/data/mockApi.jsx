// --- FILE: src/data/mockApi.jsx ---
import { mockUser, mockBooks, mockBookContent } from "./mockData"

export const mockApi = {
    login: (u, p) => new Promise(r => setTimeout(() => r(mockUser), 500)),
    fetchHomePageData: () => new Promise(r => setTimeout(() => r({
        bannerBook: mockBooks['10'],
        featured: Object.values(mockBooks).slice(0, 3),
        subscriptionOnly: Object.values(mockBooks).filter(b => b.access === 'subscription'),
        purchaseOnly: Object.values(mockBooks).filter(b => b.access === 'purchase'),
    }), 800)),
    fetchBookDetails: (id) => new Promise(r => setTimeout(() => r(mockBooks[id]), 300)),
    searchBooks: (query) => new Promise(r => setTimeout(() => {
        if (!query) return r([]);
        const lowerQuery = query.toLowerCase();
        r(Object.values(mockBooks).filter(b => b.title.toLowerCase().includes(lowerQuery) || b.author.toLowerCase().includes(lowerQuery)));
    }, 400)),
    fetchBookContent: (id) => new Promise(r => setTimeout(() => r(mockBookContent[id] || { title: mockBooks[id]?.title, paragraphs: [{text: 'Content not available.', audioTags: {age:'neutral', sense:'neutral'}}] }), 400)),
    fetchAudioForTags: (age, sense) => new Promise(r => {
        console.log("fetching: " + sense + " " + age);
        const soundMap = {
            'city_traffic': 'https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3', // Placeholder
        };
        setTimeout(() => r(soundMap[sense] || null), 200);
    }),
    fetchAudioForEntityType: (type) => new Promise(r => {
        console.log("fetching entity: " + type);
        const soundMap = { 'vehicle': 'https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3', 'emotion': 'https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3', 'vehicle_sound': 'https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3', 'weather': 'https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3', 'animal_sound': 'https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3', 'dino': 'https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3' };
        setTimeout(() => r(soundMap[type.toLowerCase()] || null), 200);
    }),
};