// --- FILE: src/pages/PaymentStatusPage.jsx (NEW FILE) ---
import React, { useEffect, useState } from 'react';
import { CheckCircleIcon } from '../components/common/icons';

const PaymentStatusPage = ({ setPage, refetchUser }) => {
    const [status, setStatus] = useState('processing'); // 'processing', 'success', 'failed'
    const [message, setMessage] = useState('Verifying your payment...');

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const paymentStatus = params.get('Status');
        const authority = params.get('Authority');

        if (paymentStatus && authority) {
            if (paymentStatus === 'OK') {
                setStatus('success');
                setMessage('Your payment was successful and your subscription is now active!');
                refetchUser();
            } else {
                setStatus('failed');
                setMessage('Your payment was cancelled or failed. Please try again.');
            }
        } else {
            setStatus('failed');
            setMessage('Invalid payment verification details.');
        }

    }, [refetchUser]);

    const renderIcon = () => {
        switch (status) {
            case 'success':
                return <CheckCircleIcon className="w-16 h-16 text-green-500" />;
            case 'failed':
                return (
                    <svg className="w-16 h-16 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                );
            default: // processing
                return (
                    <div className="w-16 h-16 border-4 border-[var(--color-accent)] border-dashed rounded-full animate-spin"></div>
                );
        }
    };


    return (
        <div className="flex items-center justify-center min-h-[calc(100vh-200px)]">
            <div className="max-w-md w-full text-center p-8 bg-[var(--color-background-secondary)] rounded-lg shadow-lg">
                <div className="flex justify-center mb-6">
                    {renderIcon()}
                </div>
                <h1 className={`text-2xl font-bold mb-2 ${status === 'success' ? 'text-green-600' : status === 'failed' ? 'text-red-600' : ''}`}>
                    {status === 'success' ? 'Payment Successful!' : status === 'failed' ? 'Payment Failed' : 'Processing Payment'}
                </h1>
                <p className="text-[var(--color-text-muted)] mb-8">{message}</p>
                <button onClick={() => setPage('profile')} className="w-full py-2 rounded-lg bg-[var(--color-accent)] text-white font-bold hover:bg-[var(--color-accent-hover)]">
                    Go to My Profile
                </button>
            </div>
        </div>
    );
};

export default PaymentStatusPage;