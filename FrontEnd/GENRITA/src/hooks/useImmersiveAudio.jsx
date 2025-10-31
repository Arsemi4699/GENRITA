// --- FILE: src/hooks/useImmersiveAudio.jsx (REVISED) ---
import { useEffect, useRef, useCallback } from 'react';
import { api } from '../services/api'; // Changed from mockApi

export const useImmersiveAudio = ({ scrollContainerRef, paragraphs, isMuted }) => {
    const paragraphPlayers = useRef(new Map());
    const entityPlayers = useRef(new Map());
    const paragraphElements = useRef(new Map());
    const triggeredParagraphs = useRef(new Set());
    const isEntitySequencePlaying = useRef(false);

    useEffect(() => {
        return () => {
            paragraphPlayers.current.forEach(p => p.player?.dispose());
            entityPlayers.current.forEach(p => p.player?.dispose());
            paragraphPlayers.current.clear();
            entityPlayers.current.clear();
        };
    }, []);

    const onScroll = useCallback(() => {
        if (isMuted || !scrollContainerRef.current) {
            paragraphPlayers.current.forEach(p => { if (p.player?.state === 'started') p.player.stop(); });
            isEntitySequencePlaying.current = false;
            return;
        }

        const container = scrollContainerRef.current;
        const { top: cTop, height: cHeight } = container.getBoundingClientRect();
        const refLine = cTop + cHeight / 4;
        const audibleDist = cHeight * 0.75;

        const activeParagraphs = [];
        paragraphElements.current.forEach((el, index) => {
            if (!el) return;
            const { top, height } = el.getBoundingClientRect();
            const pCenter = top + height / 2;
            const distance = Math.abs(pCenter - refLine);
            if (distance < audibleDist) activeParagraphs.push({ index, distance });
        });
        activeParagraphs.sort((a, b) => a.distance - b.distance);
        const closestParagraph = activeParagraphs[0];

        // Atmospheric Sound Logic
        paragraphPlayers.current.forEach((p, index) => {
            if (index !== closestParagraph?.index && p.player?.state === 'started') {
                p.player.volume.rampTo(-Infinity, 0.5);
            }
        });

        // --- Entity Player Stop Control (NEW) ---
        entityPlayers.current.forEach((e, key) => {
            const [pIndex] = key.split('-').map(Number);
            const pEl = paragraphElements.current.get(pIndex);
            if (!pEl) return;

            const { top, height } = pEl.getBoundingClientRect();
            const pCenter = top + height / 2;
            const distance = Math.abs(pCenter - refLine);

            // Stop entity if it's too far (same audibleDist threshold as paragraphs)
            if (distance > audibleDist && e.player?.state === 'started') {
                e.player.stop();
            }
        });

        if (closestParagraph) {
            const { index, distance } = closestParagraph;
            const { age, sense } = paragraphs[index].audioTags;
            if (age !== 'neutral' && sense !== 'neutral') {
                (async () => {
                    let pState = paragraphPlayers.current.get(index);
                    if (!pState) {
                        paragraphPlayers.current.set(index, { status: 'fetching', player: null });
                        try {
                            // Using the real API
                            const url = await api.fetchAudioForTags(age, sense);
                            if (!url) { paragraphPlayers.current.set(index, { status: 'failed' }); return; }
                            const player = new window.Tone.Player({ url, loop: true, fadeOut: 0.5, fadeIn: 0.5 }).toDestination();
                            await window.Tone.loaded();
                            paragraphPlayers.current.set(index, { status: 'ready', player });
                            pState = paragraphPlayers.current.get(index);
                        } catch (e) { paragraphPlayers.current.set(index, { status: 'failed' }); }
                    }
                    if (pState?.status === 'ready') {
                        if (pState.player.state !== 'started') pState.player.start();
                        const volRatio = 1 - (distance / audibleDist);
                        pState.player.volume.rampTo(window.Tone.gainToDb(Math.pow(volRatio, 2)), 0.1);
                    }
                })();
            }
        }

        // Entity Sound Logic
        const prefetchWindow = closestParagraph ? [closestParagraph.index, closestParagraph.index + 1, closestParagraph.index + 2] : [];
        prefetchWindow.forEach(async (index) => {
            if (!paragraphs[index] || !paragraphs[index].entities) return;
            for (const entity of paragraphs[index].entities) {
                const entityKey = `${index}-${entity.start_pos}`;
                if (!entityPlayers.current.has(entityKey)) {
                    entityPlayers.current.set(entityKey, { status: 'fetching', player: null });
                    try {
                        // Using the real API
                        const url = await api.fetchAudioForEntityType(entity.type);
                        if (url) {
                            const player = new window.Tone.Player({ url, loop: false, fadeIn: 0.2, fadeOut: 0.5 }).toDestination();
                            await window.Tone.loaded();
                            entityPlayers.current.set(entityKey, { status: 'ready', player });
                        } else {
                            entityPlayers.current.set(entityKey, { status: 'failed' });
                        }
                    } catch (e) { entityPlayers.current.set(entityKey, { status: 'failed' }); }
                }
            }
        });

        if (closestParagraph && closestParagraph.distance < 50 && !isEntitySequencePlaying.current) {
            if (!triggeredParagraphs.current.has(closestParagraph.index)) {
                triggeredParagraphs.current.add(closestParagraph.index);
                const paragraphData = paragraphs[closestParagraph.index];
                if (paragraphData.entities && paragraphData.entities.length > 0) {
                    isEntitySequencePlaying.current = true;
                    const sortedEntities = [...paragraphData.entities].sort((a, b) => a.start_pos - b.start_pos);
                    const sequenceQueue = sortedEntities.map(entity => `${closestParagraph.index}-${entity.start_pos}`);
                    const playNextInSequence = () => {
                        if (sequenceQueue.length === 0) {
                            isEntitySequencePlaying.current = false;
                            return;
                        }
                        const nextKey = sequenceQueue.shift();
                        const entityState = entityPlayers.current.get(nextKey);
                        if (entityState?.status === 'ready') {
                            entityState.player.onstop = playNextInSequence;
                            entityState.player.start();
                        } else {
                            playNextInSequence();
                        }
                    };
                    playNextInSequence();
                }
            }
        }

        const activeIndexes = new Set(activeParagraphs.map(p => p.index));
        triggeredParagraphs.current.forEach(index => {
            if (!activeIndexes.has(index)) {
                triggeredParagraphs.current.delete(index);
            }
        });

    }, [isMuted, paragraphs, scrollContainerRef]);

    useEffect(() => {
        const container = scrollContainerRef.current;
        if (!container) return;
        container.addEventListener('scroll', onScroll, { passive: true });
        return () => container.removeEventListener('scroll', onScroll);
    }, [onScroll]);

    const registerParagraphElement = useCallback((index, element) => paragraphElements.current.set(index, element), []);

    return { registerParagraphElement };
};
