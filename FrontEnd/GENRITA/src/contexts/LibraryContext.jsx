// --- FILE: src/contexts/LibraryContext.jsx ---
const LibraryContext = createContext();
export const useLibrary = () => useContext(LibraryContext);

export const LibraryProvider = ({ children }) => {
    const { user } = useAuth();
    const [library, setLibrary] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        if (user) {
            setIsLoading(true);
            mockApi.fetchUserLibrary(user.id).then(data => {
                setLibrary(data);
                setIsLoading(false);
            });
        } else {
            setLibrary(null);
        }
    }, [user]);

    const saveLibraryData = useCallback(async (newLibraryData) => {
        if (user) {
            await mockApi.saveLibrary(user.id, newLibraryData);
        }
    }, [user]);

    const addBookToLibrary = (bookId) => {
        setLibrary(prev => {
            const newLibrary = { ...prev };
            newLibrary.books[bookId] = { progress: 0 };
            if (!newLibrary.shelves['to-read'].bookIds.includes(bookId)) {
                newLibrary.shelves['to-read'].bookIds.push(bookId);
            }
            saveLibraryData(newLibrary);
            return newLibrary;
        });
    };

    const updateBookProgress = (bookId, progress) => {
        setLibrary(prev => {
            const newLibrary = { ...prev };
            if(newLibrary.books[bookId]) {
                newLibrary.books[bookId].progress = progress;
            }
            // Logic to move book to 'finished' shelf if progress is 1
            if(progress >= 1 && !newLibrary.shelves.finished.bookIds.includes(bookId)){
                // Remove from other shelves
                Object.keys(newLibrary.shelves).forEach(shelfId => {
                    newLibrary.shelves[shelfId].bookIds = newLibrary.shelves[shelfId].bookIds.filter(id => id !== bookId);
                });
                newLibrary.shelves.finished.bookIds.push(bookId);
            }
            saveLibraryData(newLibrary);
            return newLibrary;
        });
    };

    const value = { library, isLoading, addBookToLibrary, updateBookProgress };
    return <LibraryContext.Provider value={value}>{children}</LibraryContext.Provider>;
};