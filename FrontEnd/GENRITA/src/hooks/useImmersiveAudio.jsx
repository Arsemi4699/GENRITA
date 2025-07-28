import { useEffect, useRef, useCallback } from 'react';
import { mockApi } from '../data/mockApi';

export const useImmersiveAudio = ({ scrollContainerRef, paragraphs, isMuted }) => {
    const paragraphPlayers = useRef(new Map());
    const entityPlayers = useRef(new Map());
    const paragraphElements = useRef(new Map());
    const triggeredParagraphs = useRef(new Set());
    const isEntitySequencePlaying = useRef(false); // Lock for entity sequences only

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
            isEntitySequencePlaying.current = false; // Reset lock on mute
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

        // --- 1. Atmospheric Sound Logic (Looping, Independent) ---
        paragraphPlayers.current.forEach((p, index) => {
            if (index !== closestParagraph?.index && p.player?.state === 'started') {
                p.player.volume.rampTo(-Infinity, 0.5);
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
                            const url = await mockApi.fetchAudioForTags(age, sense);
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

        // --- 2. Entity Sound Logic (Queued within paragraph, one-shot) ---
        const prefetchWindow = closestParagraph ? [closestParagraph.index, closestParagraph.index + 1, closestParagraph.index + 2] : [];
        prefetchWindow.forEach(async (index) => {
            if (!paragraphs[index] || !paragraphs[index].entities) return;
            for (const entity of paragraphs[index].entities) {
                const entityKey = `${index}-${entity.start_pos}`;
                if (!entityPlayers.current.has(entityKey)) {
                    entityPlayers.current.set(entityKey, { status: 'fetching', player: null });
                    try {
                        const url = await mockApi.fetchAudioForEntityType(entity.type);
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

        // Trigger sequence only if the lock is free
        if (closestParagraph && closestParagraph.distance < 50 && !isEntitySequencePlaying.current) {
            if (!triggeredParagraphs.current.has(closestParagraph.index)) {
                triggeredParagraphs.current.add(closestParagraph.index);
                const paragraphData = paragraphs[closestParagraph.index];
                if (paragraphData.entities && paragraphData.entities.length > 0) {

                    isEntitySequencePlaying.current = true; // Acquire the lock

                    const sortedEntities = [...paragraphData.entities].sort((a, b) => a.start_pos - b.start_pos);
                    const sequenceQueue = sortedEntities.map(entity => `${closestParagraph.index}-${entity.start_pos}`);

                    const playNextInSequence = () => {
                        if (sequenceQueue.length === 0) {
                            isEntitySequencePlaying.current = false; // Release the lock
                            return;
                        }
                        const nextKey = sequenceQueue.shift();
                        const entityState = entityPlayers.current.get(nextKey);
                        if (entityState?.status === 'ready') {
                            entityState.player.onstop = playNextInSequence;
                            entityState.player.start();
                        } else {
                            playNextInSequence(); // Skip if not ready and play the next one
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
