// --- FILE: src/pages/ReaderPage.js ---
import { useState, useEffect, useRef } from 'react';
import { mockApi } from '../data/mockApi';
import { BackIcon, VolumeOffIcon, VolumeOnIcon } from '../components/common/icons/index';
import { useAudioSettings } from '../contexts/AudioSettingsContext';
import { useImmersiveAudio } from '../hooks/useImmersiveAudio';

const ReaderPage = ({ bookId, setPage }) => {
    const [content, setContent] = useState(null);
    const scrollContainerRef = useRef(null);
    const audioSettings = useAudioSettings();

    const isMuted = audioSettings ? audioSettings.isMuted : true;
    const toggleMute = audioSettings ? audioSettings.toggleMute : () => { };

    // NOTE: registerWordElement is no longer needed as we don't track individual words
    const { registerParagraphElement } = useImmersiveAudio({
        scrollContainerRef,
        paragraphs: content?.paragraphs || [],
        isMuted
    });

    useEffect(() => {
        setContent(null);
        mockApi.fetchBookContent(bookId).then(setContent);
    }, [bookId]);

    // NEW: Helper function to parse and render text with entity highlights
    const renderParagraphWithEntities = (paragraph) => {
        if (!paragraph.entities || paragraph.entities.length === 0) {
            return paragraph.text;
        }

        const sortedEntities = [...paragraph.entities].sort((a, b) => a.start_pos - b.start_pos);
        const parts = [];
        let lastIndex = 0;

        sortedEntities.forEach(entity => {
            // Add text before the entity
            parts.push(paragraph.text.substring(lastIndex, entity.start_pos));
            // Add the highlighted entity
            parts.push(
                <span key={entity.start_pos} className="font-bold text-[var(--color-accent)]">
                    {entity.sample}
                </span>
            );
            lastIndex = entity.start_pos + entity.sample.length;
        });

        // Add the remaining text after the last entity
        parts.push(paragraph.text.substring(lastIndex));

        return parts;
    };

    if (!content) return <div className="p-6 text-center">Loading content...</div>;

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
                <div className="text-[var(--color-text-primary)] text-lg leading-relaxed">
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    <br></br>
                    <h1 className="text-4xl font-bold mb-8 border-b border-[var(--color-border)] pb-4">{content.title}</h1>
                    {content.paragraphs.map((p, i) => (
                        <p key={i} ref={el => registerParagraphElement(i, el)} className="mb-6">
                            {renderParagraphWithEntities(p)}
                        </p>
                    ))}
                </div>
            </div>
        </div>
    );
}
export default ReaderPage;