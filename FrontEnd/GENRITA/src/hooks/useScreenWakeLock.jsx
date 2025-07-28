// --- FILE: src/hooks/useScreenWakeLock.js ---
const useScreenWakeLock = () => {
    const wakeLock = useRef(null);

    const requestWakeLock = async () => {
        if ('wakeLock' in navigator) {
            try {
                wakeLock.current = await navigator.wakeLock.request('screen');
                console.log('Screen Wake Lock is active.');
            } catch (err) {
                console.error(`${err.name}, ${err.message}`);
            }
        }
    };

    const releaseWakeLock = () => {
        if (wakeLock.current) {
            wakeLock.current.release().then(() => {
                wakeLock.current = null;
                console.log('Screen Wake Lock released.');
            });
        }
    };

    useEffect(() => {
        requestWakeLock();
        return () => releaseWakeLock();
    }, []);
};