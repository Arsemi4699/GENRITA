// --- FILE: src/services/api.jsx (REVISED) ---
import axios from 'axios';

const API_BASE_URL = 'https://eivazi.qzz.io';

// THE FIX: Add the 'export' keyword here so other files can import it.
export const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: { 'Content-Type': 'application/json' },
});

apiClient.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('accessToken');
        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
    (response) => response,
    async (error) => {
        const originalRequest = error.config;
        if (error.response?.status === 401 && !originalRequest._retry) {
            originalRequest._retry = true;
            try {
                const refreshToken = localStorage.getItem('refreshToken');
                if (!refreshToken) throw new Error("No refresh token available");
                // Note: The logout function is on the 'api' object which isn't defined yet.
                // We'll define a standalone logout helper.
                const standaloneLogout = () => {
                    localStorage.removeItem('accessToken');
                    localStorage.removeItem('refreshToken');
                };
                console.log("sending refresh requset");
                const { data } = await axios.post(`${API_BASE_URL}/Auth/Refresh`, { refreshToken });
                localStorage.setItem('accessToken', data.accessToken);
                apiClient.defaults.headers.common['Authorization'] = `Bearer ${data.accessToken}`;
                return apiClient(originalRequest);
            } catch (refreshError) {
                // Use the standalone helper here
                localStorage.removeItem('accessToken');
                localStorage.removeItem('refreshToken');
                window.location.replace('/');
                return Promise.reject(refreshError);
            }
        }
        return Promise.reject(error);
    }
);

export const api = {
    // Both register and login now send the role
    register: async (name, email, password, role) => {
        const { data } = await apiClient.post('/Auth/Register', { name, email, password, role });
        return data;
    },
    login: async (email, password, role) => {
        const { data } = await apiClient.post('/Auth/Login', { email, password, role });
        localStorage.setItem('accessToken', data.accessToken);
        localStorage.setItem('refreshToken', data.refreshToken);
        return data;
    },
    logout: () => {
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
    },
    // fetchMe correctly gets the persistent role from the backend
    fetchMe: async () => {
        const { data } = await apiClient.get('/Auth/Me');
        return data;
    },
    fetchHomePageData: async () => {
        const { data } = await apiClient.get('/api/Home');
        const adaptBook = (b) => ({ ...b, author: b.authorName || 'Unknown Author' });
        return {
            bannerBook: data.bannerBook ? adaptBook(data.bannerBook) : null,
            featured: data.featured?.map(adaptBook) || [],
            subscriptionOnly: data.subscriptionOnly?.map(adaptBook) || [],
            freeOnly: data.freeOnly?.map(adaptBook) || [],
        };
    },
    fetchBookDetails: async (id) => {
        const { data } = await apiClient.get(`/books/${id}`);
        return { ...data, author: data.author || 'Unknown Author' };
    },
    fetchBookContent: async (id) => {
        const { data } = await apiClient.get(`/books/${id}/content`);
        return data;
    },
    searchBooks: async (query, params = {}) => {
        const finalParams = { ...params };
        if (!finalParams.publishedDate) delete finalParams.publishedDate;
        if (finalParams.searchMode === 0) delete finalParams.searchMode;
        if (finalParams.order === 0) delete finalParams.order;
        const { data } = await apiClient.post(`/books/search?q=${encodeURIComponent(query)}`, finalParams);
        return Array.isArray(data) ? data.map(b => ({ ...b, author: b.authorName || 'Unknown Author' })) : [];
    },
    addToLibrary: async (bookId) => {
        const { data } = await apiClient.post('/Users/Library', { bookId });
        return data;
    },
    removeFromLibrary: async (bookId) => {
        const { data } = await apiClient.delete(`/Users/Library/${bookId}`);
        return data;
    },
    fetchAudioForTags: async (age, sense) => {
        try {
            const { data } = await apiClient.get(`/Audio/Audio/tags?age=${age}&sense=${sense}`);
            console.log("request for class: " + age + " " + sense);
            return data.url;
        } catch (error) { return null; }
    },
    fetchAudioForEntityType: async (type) => {
        try {
            const { data } = await apiClient.get(`/Audio/Audio/entity?type=${type}`);
            console.log("request for entity: " + type);
            return data.url;
        } catch (error) { return null; }
    },
    // --- NEW ---
    getAllPlans: async () => {
        const { data } = await apiClient.get('/Plans/GetAllPlans');
        return data;
    },
    createPayment: async (planId) => {
        const { data } = await apiClient.post(`/Transaction/CreatePayment?planId=${planId}`);
        return data; // Expecting the payment URL string
    },
};