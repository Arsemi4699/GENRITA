// --- FILE: src/services/authorApi.js (REVISED AND FINAL) ---
// This file is now rewritten to use the real apiClient and connect to the backend.
import { apiClient } from './api';

export const authorApi = {
    /**
     * Fetches the list of books/jobs for the logged-in author.
     */
    listAuthorBooks: async () => {
        console.log("FRONTEND: Requesting author's book list from REAL API.");
        const { data } = await apiClient.get('/author/books');
        console.log(data);
        return data;
    },

    /**
     * Submits a new book with all its metadata and AI config to our backend.
     * @param {FormData} formData The complete form data object.
     */
    processNewBook: async (formData) => {
        console.log("FRONTEND: Submitting new book form data...");
        console.log(formData);
        // We must override the default 'Content-Type' header for file uploads.
        // The browser will automatically set the correct boundary.
        const { data } = await apiClient.post('/author/books/process', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        return data; // Returns { jobId, message } from the backend
    },

    /**
     * Fetches the status of a specific job from our backend.
     */
    getJobStatus: async (jobId) => {
        console.log(`FRONTEND: Polling status for job ${jobId}`);
        const { data } = await apiClient.get(`/author/jobs/${jobId}`);
        console.log(data);
        return data;
    },

    /**
     * Fetches all available book categories from the backend.
     * Endpoint: GET /books/categories/getall
     */
    getAllCategories: async () => {
        console.log("FRONTEND: Fetching book categories from REAL API...");
        const { data } = await apiClient.get('/books/categories/getall');
        return data; // Expecting [{ id, categoryName }, ...]
    },

    deleteBook: async (id) => {
        console.log(`FRONTEND: Deleting book with ID ${id}`);
        const { data } = await apiClient.delete(`/books/${id}`);
        return data;
    },
};