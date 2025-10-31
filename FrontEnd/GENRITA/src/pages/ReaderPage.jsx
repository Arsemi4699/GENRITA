

// --- FILE: src/pages/ReaderPage.jsx (REVISED) ---
import { useState, useEffect, useRef } from 'react';
import { api } from '../services/api'; // Changed from mockApi
import { BackIcon, VolumeOffIcon, VolumeOnIcon } from '../components/common/icons/index';
import { useAudioSettings } from '../contexts/AudioSettingsContext';
import { useImmersiveAudio } from '../hooks/useImmersiveAudio';

const ReaderPage = ({ bookId, setPage }) => {
    const [content, setContent] = useState(null);
    const scrollContainerRef = useRef(null);
    const { isMuted, toggleMute } = useAudioSettings();
    const [book, setBook] = useState(null);

    useEffect(() => {
        setBook(null);
        api.fetchBookDetails(bookId).then(setBook).catch(console.error);
    }, [bookId]);

    const { registerParagraphElement } = useImmersiveAudio({
        scrollContainerRef,
        paragraphs: content?.paragraphs || [],
        isMuted
    });

    useEffect(() => {
        setContent(null);
        // Changed from mockApi.fetchBookContent to api.fetchBookContent
        api.fetchBookContent(bookId).then(setContent).catch(err => console.error("Failed to load content", err));
    }, [bookId]);

    const renderParagraphWithEntities = (paragraph) => {
        if (!paragraph.entities || paragraph.entities.length === 0) {
            return paragraph.text;
        }
        const sortedEntities = [...paragraph.entities].sort((a, b) => a.start_pos - b.start_pos);
        const parts = [];
        let lastIndex = 0;
        sortedEntities.forEach(entity => {
            parts.push(paragraph.text.substring(lastIndex, entity.start_pos));
            parts.push(
                <span key={entity.start_pos} className="font-bold text-[var(--color-accent)]">
                    {entity.sample}
                </span>
            );
            lastIndex = entity.start_pos + entity.sample.length;
        });
        parts.push(paragraph.text.substring(lastIndex));
        return parts;
    };

    if (!content) return <div className="p-6 text-center">Loading content...</div>;
    console.log(content.paragraphs);
    return (
        <div ref={scrollContainerRef} className="fixed inset-0 bg-[var(--color-background)] overflow-y-auto">
            <div className="sticky top-0 z-10 flex items-center justify-between p-2 bg-[var(--color-background)]/80 backdrop-blur-sm">
                <button onClick={() => setPage('bookDetails')} className="p-2 text-[var(--color-text-primary)] rounded-full hover:bg-[var(--color-background-secondary)]">
                    <BackIcon />
                </button>
                <button onClick={toggleMute} className="p-2 text-[var(--color-text-primary)] rounded-full hover:bg-[var(--color-background-secondary)]">
                    {isMuted ? <VolumeOffIcon /> : <VolumeOnIcon />}
                </button>
            </div>
            <div className="max-w-3xl mx-auto px-6 pb-12">
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <div className="max-w-4xl mx-auto md:flex md:space-x-8">
                    <div className="md:w-1/3 mb-6 md:mb-0">
                        <img src={book.cover} alt={book.title} className="rounded-lg shadow-lg w-full" />
                    </div>
                    <div className="md:w-2/3 text-[var(--color-text-primary)]">
                        <h1 className="text-3xl md:text-4xl font-bold">{book.title}</h1>
                        <p className="text-lg text-[var(--color-text-muted)] mt-1">{book.authorName}</p>
                    </div>
                </div>
                <div className="text-[var(--color-text-primary)] text-lg leading-relaxed">
                    <h1 className="text-4xl font-bold mb-8 border-b border-[var(--color-border)] pb-4">{content.title}</h1>
                    {content.paragraphs.map((p, i) => (
                        <p key={i} ref={el => registerParagraphElement(i, el)} className="mb-6">
                            {renderParagraphWithEntities(p)}
                        </p>
                    ))}
                </div>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
            </div>
        </div>
    );
}
export default ReaderPage;

