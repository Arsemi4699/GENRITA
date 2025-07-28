import { useState, createContext, useContext, useCallback } from 'react';

const AudioSettingsContext = createContext(null);
export const useAudioSettings = () => useContext(AudioSettingsContext);

export const AudioSettingsProvider = ({ children }) => {
    const [isMuted, setIsMuted] = useState(true);
    const [isAudioReady, setIsAudioReady] = useState(false);

    const startAudio = useCallback(async () => {
        if (isAudioReady) return;
        if (window.Tone && window.Tone.context.state !== 'running') {
            await window.Tone.start();
            console.log("AudioContext started successfully!");
            setIsAudioReady(true);
        }
    }, [isAudioReady]);

    const toggleMute = useCallback(() => {
        // Start audio on first unmute if it hasn't started yet
        if (isMuted) {
            startAudio();
        }
        setIsMuted(prev => !prev);
    }, [isMuted, startAudio]);

    return (
        <AudioSettingsContext.Provider value={{ isMuted, toggleMute, startAudio }}>
            {children}
        </AudioSettingsContext.Provider>
    );
};
