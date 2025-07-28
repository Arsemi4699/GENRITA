// --- FILE: src/pages/HomePage.jsx (REVISED) ---
import { useState, useEffect } from 'react';
import { mockApi } from '../data/mockApi';
import { HomePageSkeleton } from '../components/common/SkeletonLoader';
import Carousel from '../components/common/Carousel';
import BookCard from '../components/common/BookCard';

const HomePage = ({ setPage, setSelectedBookId }) => {
    const [homeData, setHomeData] = useState(null);
    useEffect(() => {
        // Simulate a slightly longer loading time to see the skeleton
        setTimeout(() => {
            mockApi.fetchHomePageData().then(setHomeData);
        }, 1000);
    }, []);
    const viewDetails = (bookId) => { setSelectedBookId(bookId); setPage('bookDetails'); };

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            {!homeData ? (
                <HomePageSkeleton />
            ) : (
                <div className="py-6">
                    <div className="mb-10">
                        <div className="relative rounded-lg overflow-hidden p-8 bg-cover bg-center min-h-[200px] flex items-end" style={{ backgroundImage: `url(${homeData.bannerBook.cover})` }}>
                            <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent"></div>
                            <div className="relative z-10">
                                <h2 className="text-white text-3xl font-bold">{homeData.bannerBook.title}</h2>
                                <p className="text-white/80">{homeData.bannerBook.author}</p>
                                <button onClick={() => viewDetails(homeData.bannerBook.id)} className="mt-4 bg-white text-black px-4 py-2 rounded-lg font-semibold text-sm">View Details</button>
                            </div>
                        </div>
                    </div>
                    <Carousel title="Featured">
                        {homeData.featured.map(b => <div key={b.id} className="w-36 md:w-40 flex-shrink-0"><BookCard book={b} onClick={() => viewDetails(b.id)} /></div>)}
                    </Carousel>
                    <Carousel title="Available with Subscription">
                        {homeData.subscriptionOnly.map(b => <div key={b.id} className="w-36 md:w-40 flex-shrink-0"><BookCard book={b} onClick={() => viewDetails(b.id)} /></div>)}
                    </Carousel>
                    <Carousel title="Available for Purchase">
                        {homeData.purchaseOnly.map(b => <div key={b.id} className="w-36 md:w-40 flex-shrink-0"><BookCard book={b} onClick={() => viewDetails(b.id)} /></div>)}
                    </Carousel>
                </div>
            )}
        </div>
    );
}
export default HomePage;