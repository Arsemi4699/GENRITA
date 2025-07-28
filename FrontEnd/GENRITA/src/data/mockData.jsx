// --- FILE: src/data/mockData.jsx ---

export const mockBooks = {
    '1': { id: '1', title: 'Blindness', author: 'José Saramago', category: 'Fiction', cover: 'https://placehold.co/300x450/a16207/FFFFFF?text=Blindness', access: 'subscription', synopsis: 'A city is struck by an epidemic of "white blindness" which spares no one.' },
    '10': { id: '10', title: 'Dune', author: 'Frank Herbert', category: 'Sci-Fi', cover: 'https://placehold.co/300x450/c26a37/FFFFFF?text=Dune', access: 'subscription', synopsis: 'The story of young Paul Atreides in a distant future feudal interstellar society.' },
    '11': { id: '11', title: 'Atomic Habits', author: 'James Clear', category: 'Self-Help', cover: 'https://placehold.co/300x450/f2cc8f/000000?text=Atomic+Habits', access: 'purchase', synopsis: 'An easy & proven way to build good habits & break bad ones.' },
    '4': { id: '4', title: 'The Little Prince', author: 'Antoine de Saint-Exupéry', category: 'Children', cover: 'https://placehold.co/300x450/fca311/FFFFFF?text=The+Little+Prince', access: 'free', synopsis: 'A young prince who visits various planets in space, including Earth.' },
};
export const mockBookContent = {
    '1': {
        title: 'Blindness',
        paragraphs: [
            {
                text: 'He cried out, "I am blind!" his voice a howl of terror. Panic rippled through the other drivers as the car swerved erratically, coming to a halt in the middle of the road. Horns blared. Tires screeched. A woman in the next lane leaned out of her window, shouting, "Are you okay? Whats happening? But the man didn’t respond. He clutched the steering wheel with white-knuckled fingers, eyes wide and unfocused. His chest heaved with each breath. He reached for the door handle, fumbling. Then another voice rose—this time from the sidewalk. A passerby had collapsed, clutching their face. “I can’t see! I can’t see anything!” they wailed. In moments, it became clear this wasn’t an isolated incident. People began stumbling in all directions—drivers, pedestrians, cyclists—blinded in an instant, their cries overlapping in a growing chorus of confusion and fear. From the chaos, a single word echoed louder than the rest: “Why?” He cried out, "I am blind!" his voice a howl of terror. Panic rippled through the other drivers as the car swerved erratically, coming to a halt in the middle of the road. Horns blared. Tires screeched. A woman in the next lane leaned out of her window, shouting, "Are you okay? Whats happening? But the man didn’t respond. He clutched the steering wheel with white-knuckled fingers, eyes wide and unfocused. His chest heaved with each breath. He reached for the door handle, fumbling. Then another voice rose—this time from the sidewalk. A passerby had collapsed, clutching their face. “I can’t see! I can’t see anything!” they wailed. In moments, it became clear this wasn’t an isolated incident. People began stumbling in all directions—drivers, pedestrians, cyclists—blinded in an instant, their cries overlapping in a growing chorus of confusion and fear. From the chaos, a single word echoed louder than the rest: “Why?” Suddenly, one of the cars stopped. The driver had gone blind. He cried out, "I am blind!" his voice a howl of terror. Panic rippled through the other drivers as the car swerved erratically, coming to a halt in the middle of the road. Horns blared. Tires screeched. A woman in the next lane leaned out of her window, shouting, "Are you okay? Whats happening? But the man didn’t respond. He clutched the steering wheel with white-knuckled fingers, eyes wide and unfocused. His chest heaved with each breath. He reached for the door handle, fumbling. Then another voice rose—this time from the sidewalk. A passerby had collapsed, clutching their face. “I can’t see! I can’t see anything!” they wailed. In moments, it became clear this wasn’t an isolated incident. People began stumbling in all directions—drivers, pedestrians, cyclists—blinded in an instant, their cries overlapping in a growing chorus of confusion and fear. From the chaos, a single word echoed louder than the rest: “Why?” He cried out, "I am blind!" his voice a howl of terror. Panic rippled through the other drivers as the car swerved erratically, coming to a halt in the middle of the road. Horns blared. Tires screeched. A woman in the next lane leaned out of her window, shouting, "Are you okay? Whats happening? But the man didn’t respond. He clutched the steering wheel with white-knuckled fingers, eyes wide and unfocused. His chest heaved with each breath. He reached for the door handle, fumbling. Then another voice rose—this time from the sidewalk. A passerby had collapsed, clutching their face. “I can’t see! I can’t see anything!” they wailed. In moments, it became clear this wasn’t an isolated incident. People began stumbling in all directions—drivers, pedestrians, cyclists—blinded in an instant, their cries overlapping in a growing chorus of confusion and fear. From the chaos, a single word echoed louder than the rest: “Why?” He cried out, "I am blind!" his voice a howl of terror. Panic rippled through the other drivers as the car swerved erratically, coming to a halt in the middle of the road. Horns blared. Tires screeched. A woman in the next lane leaned out of her window, shouting, "Are you okay? Whats happening? But the man didn’t respond. He clutched the steering wheel with white-knuckled fingers, eyes wide and unfocused. His chest heaved with each breath. He reached for the door handle, fumbling. Then another voice rose—this time from the sidewalk. A passerby had collapsed, clutching their face. “I can’t see! I can’t see anything!” they wailed. In moments, it became clear this wasn’t an isolated incident. People began stumbling in all directions—drivers, pedestrians, cyclists—blinded in an instant, their cries overlapping in a growing chorus of confusion and fear. From the chaos, a single word echoed louder than the rest: “Why?”',
                audioTags: { age: 'modern', sense: 'panic' },
                entities: [
                    { type: "vehicle", sample: "cars", start_pos: 21 },
                    { type: "emotion", sample: "terror", start_pos: 110 }
                ]
            },
            {
                text: 'Other drivers, annoyed by the holdup, began to honk their horns. A crowd gathered.',
                audioTags: { age: 'modern', sense: 'city_traffic' },
                entities: [
                    { type: "vehicle_sound", sample: "horns", start_pos: 58 }
                ]
            },
            {
                text: 'Paul could hear the faint whisper of the wind outside, a lonely sound.',
                audioTags: { age: 'old', sense: 'neutral' },
                entities: [
                    { type: "weather", sample: "wind", start_pos: 41 }
                ]
            },
            {
                text: 'Suddenly, a great roar shook the foundations of the castle, a sound like a great T-rex awakening.',
                audioTags: { age: 'old', sense: 'battle' },
                entities: [
                    { type: "animal_sound", sample: "roar", start_pos: 18 },
                    { type: "dino", sample: "T-rex", start_pos: 105 }
                ]
            },
            {
                text: 'Suddenly, one of the cars stopped. The driver had gone blind. He cried out, "I am blind!" his voice a howl of terror.',
                audioTags: { age: 'modern', sense: 'panic' },
                entities: [
                    { type: "vehicle", sample: "cars", start_pos: 21 },
                    { type: "emotion", sample: "terror", start_pos: 110 }
                ]
            },
            {
                text: 'Other drivers, annoyed by the holdup, began to honk their horns. A crowd gathered.',
                audioTags: { age: 'modern', sense: 'city_traffic' },
                entities: [
                    { type: "vehicle_sound", sample: "horns", start_pos: 58 }
                ]
            },
            {
                text: 'Paul could hear the faint whisper of the wind outside, a lonely sound.',
                audioTags: { age: 'old', sense: 'neutral' },
                entities: [
                    { type: "weather", sample: "wind", start_pos: 41 }
                ]
            },
            {
                text: 'Suddenly, a great roar shook the foundations of the castle, a sound like a great T-rex awakening.',
                audioTags: { age: 'old', sense: 'battle' },
                entities: [
                    { type: "animal_sound", sample: "roar", start_pos: 18 },
                    { type: "dino", sample: "T-rex", start_pos: 105 }
                ]
            },
            {
                text: 'Suddenly, one of the cars stopped. The driver had gone blind. He cried out, "I am blind!" his voice a howl of terror.',
                audioTags: { age: 'modern', sense: 'panic' },
                entities: [
                    { type: "vehicle", sample: "cars", start_pos: 21 },
                    { type: "emotion", sample: "terror", start_pos: 110 }
                ]
            },
            {
                text: 'Other drivers, annoyed by the holdup, began to honk their horns. A crowd gathered.',
                audioTags: { age: 'modern', sense: 'city_traffic' },
                entities: [
                    { type: "vehicle_sound", sample: "horns", start_pos: 58 }
                ]
            },
            {
                text: 'Paul could hear the faint whisper of the wind outside, a lonely sound.',
                audioTags: { age: 'old', sense: 'neutral' },
                entities: [
                    { type: "weather", sample: "wind", start_pos: 41 }
                ]
            },
            {
                text: 'Suddenly, a great roar shook the foundations of the castle, a sound like a great T-rex awakening.',
                audioTags: { age: 'old', sense: 'battle' },
                entities: [
                    { type: "animal_sound", sample: "roar", start_pos: 18 },
                    { type: "dino", sample: "T-rex", start_pos: 105 }
                ]
            },
            {
                text: 'Suddenly, one of the cars stopped. The driver had gone blind. He cried out, "I am blind!" his voice a howl of terror.',
                audioTags: { age: 'modern', sense: 'panic' },
                entities: [
                    { type: "vehicle", sample: "cars", start_pos: 21 },
                    { type: "emotion", sample: "terror", start_pos: 110 }
                ]
            },
            {
                text: 'Other drivers, annoyed by the holdup, began to honk their horns. A crowd gathered.',
                audioTags: { age: 'modern', sense: 'city_traffic' },
                entities: [
                    { type: "vehicle_sound", sample: "horns", start_pos: 58 }
                ]
            },
            {
                text: 'Paul could hear the faint whisper of the wind outside, a lonely sound.',
                audioTags: { age: 'old', sense: 'neutral' },
                entities: [
                    { type: "weather", sample: "wind", start_pos: 41 }
                ]
            },
            {
                text: 'Suddenly, a great roar shook the foundations of the castle, a sound like a great T-rex awakening.',
                audioTags: { age: 'old', sense: 'battle' },
                entities: [
                    { type: "animal_sound", sample: "roar", start_pos: 18 },
                    { type: "dino", sample: "T-rex", start_pos: 105 }
                ]
            },
        ]
    },
    '10': {
        title: 'Dune',
        paragraphs: [
            {
                text: 'Paul could hear the faint whisper of the wind outside, a lonely sound.',
                audioTags: { age: 'old', sense: 'neutral' },
                entities: [
                    { type: "weather", sample: "wind", start_pos: 41 }
                ]
            },
            {
                text: 'Suddenly, a great roar shook the foundations of the castle, a sound like a great T-rex awakening.',
                audioTags: { age: 'old', sense: 'battle' },
                entities: [
                    { type: "animal_sound", sample: "roar", start_pos: 18 },
                    { type: "dino", sample: "T-rex", start_pos: 105 }
                ]
            },
        ]
    },
};
export const mockUser = {
    id: 'user123', name: 'Alex', email: 'alex@example.com',
    subscription: { status: 'active', expiryDate: '2025-12-31' },
    ownedBookIds: ['11'],
    libraryBookIds: ['1', '4', '10', '11'],
};