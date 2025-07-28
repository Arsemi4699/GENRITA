// --- FILE: src/pages/StorePage.jsx ---
import { useState, useEffect } from 'react';

function StorePage({ setPage, setSelectedBookId }) {
    const [storeData, setStoreData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const { library, addBookToLibrary } = useLibrary();

    useEffect(() => {
        mockApi.fetchStoreData().then(data => {
            setStoreData(data);
            setIsLoading(false);
        });
    }, []);

    const handleBookClick = (bookId) => {
        setSelectedBookId(bookId);
        // Here you could open a book detail modal/page
        if (!library.books[bookId]) {
            addBookToLibrary(bookId);
            // TODO: Show a toast notification: "Added to library!"
        }
    };

    if (isLoading) {
        return (
            <div className="p-4 md:p-6">
                <div className="h-8 bg-gray-300 dark:bg-gray-700 rounded w-1/3 mb-6 animate-pulse"></div>
                <div className="flex space-x-4 overflow-hidden">
                    {[...Array(3)].map((_, i) => <div key={i} className="w-40 flex-shrink-0"><SkeletonLoader type="card" /></div>)}
                </div>
            </div>
        );
    }

    return (
        <div className="p-4 md:p-6 pb-20 md:pb-6">
            <Carousel title="Featured Books">
                {storeData.featured.map(book => (
                    <div key={book.id} className="w-40 flex-shrink-0">
                        <BookCard book={book} onClick={() => handleBookClick(book.id)} />
                    </div>
                ))}
            </Carousel>
            <Carousel title="New Releases">
                {storeData.newReleases.map(book => (
                    <div key={book.id} className="w-40 flex-shrink-0">
                        <BookCard book={book} onClick={() => handleBookClick(book.id)} />
                    </div>
                ))}
            </Carousel>
        </div>
    );
}