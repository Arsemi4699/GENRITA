import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { authorApi } from '../services/authorApi';
import MultiSelectDropdown from '../components/common/MultiSelectDropdown';

// --- MOCK DATA FOR FORM OPTIONS ---
const SENSE_CLASSES = {
    "Normal and neutral": 0, "Love and romantic": 1, "War and combat": 2, "Fantasy and mythology": 3,
    "Honor and respect": 4, "Drama and tragedy": 5, "City and Crowd": 6, "Mountain and the heights": 7,
    "Desert and dunes": 8, "Sea and tides": 9, "Forest and tress": 10,
};
const AGE_CLASSES = {
    "ancient and old age": 0,
    "neutral and not special age (non-ancient, non technology)": 1,
    "technology modern age": 2,
};


const NewBookPage = ({ setPage, setSelectedJobId }) => {
    const { user } = useAuth();
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [coverPreview, setCoverPreview] = useState(null);
    const [categories, setCategories] = useState([]);
    const [isCategoriesLoading, setIsCategoriesLoading] = useState(true);

    // Consolidated state for all book metadata fields
    const [bookDetails, setBookDetails] = useState({
        Title: '',
        Synopsis: '',
        Image: null,
        Access: 'Subscription',
        PublishedDate: '',
        CategoryId: "",
        file: null,
    });

    // AI config state
    const [aiConfig, setAiConfig] = useState({
        confidence_threshold: 0.85,
        allowed_senses: [],
        allowed_ages: [],
        target_abstracts: [{ name: '', description: '', file: null }],
    });

    // State for the new situational audio feature
    const [situationalAudio, setSituationalAudio] = useState([]);

    // --- Fetch categories from backend ---
    useEffect(() => {
        const fetchCategories = async () => {
            try {
                const data = await authorApi.getAllCategories();
                setCategories(data || []);
                if (data?.length > 0) {
                    setBookDetails(prev => ({ ...prev, CategoryId: data[0].id }));
                }
            } catch (err) {
                console.error("Failed to load categories:", err);
                setError("Failed to load categories from server.");
            } finally {
                setIsCategoriesLoading(false);
            }
        };
        fetchCategories();
    }, []);

    const handleDetailChange = (e) => {
        const { name, value } = e.target;
        setBookDetails(prev => ({ ...prev, [name]: value }));
    };

    const handleFileChange = (e) => {
        const { name, files } = e.target;
        if (files.length > 0) {
            const file = files[0];
            setBookDetails(prev => ({ ...prev, [name]: file }));
            if (name === 'Image') {
                setCoverPreview(URL.createObjectURL(file));
            }
        }
    };

    const handleAiConfigChange = (field, value) => {
        setAiConfig(prev => ({ ...prev, [field]: value }));
    };

    const handleAbstractChange = (index, field, value) => {
        const newAbstracts = [...aiConfig.target_abstracts];
        newAbstracts[index][field] = value;
        handleAiConfigChange('target_abstracts', newAbstracts);
    };

    const handleAbstractFileChange = (index, file) => {
        const newAbstracts = [...aiConfig.target_abstracts];
        newAbstracts[index].file = file;
        handleAiConfigChange('target_abstracts', newAbstracts);
    };

    const addAbstract = () => {
        handleAiConfigChange('target_abstracts', [...aiConfig.target_abstracts, { name: '', description: '', file: null }]);
    };

    const removeAbstract = (index) => {
        const newAbstracts = aiConfig.target_abstracts.filter((_, i) => i !== index);
        handleAiConfigChange('target_abstracts', newAbstracts);
    };

    // --- Handlers for Situational Audio ---
    const addSituationalAudio = (senseId, ageId, file) => {
        if (!senseId || !ageId || !file) {
            setError("Please select a sense, an age, and an audio file.");
            return;
        }
        // Check if the tuple already exists
        const exists = situationalAudio.some(item => item.senseId === senseId && item.ageId === ageId);
        if (exists) {
            setError("An audio file for this Sense/Age combination already exists.");
            return;
        }

        const senseName = Object.keys(SENSE_CLASSES).find(key => SENSE_CLASSES[key] == senseId);
        const ageName = Object.keys(AGE_CLASSES).find(key => AGE_CLASSES[key] == ageId);

        setSituationalAudio(prev => [...prev, { senseId, ageId, file, senseName, ageName }]);
        setError(null); // Clear previous errors
    };

    const removeSituationalAudio = (index) => {
        setSituationalAudio(prev => prev.filter((_, i) => i !== index));
    };

    const SituationalAudioInput = () => {
        const [sense, setSense] = useState("");
        const [age, setAge] = useState("");
        const [file, setFile] = useState(null);

        const handleAddClick = () => {
            addSituationalAudio(sense, age, file);
            // Reset form for next entry
            setSense("");
            setAge("");
            document.getElementById('situational-audio-file-input').value = '';
            setFile(null);
        };

        return (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end p-3 bg-[var(--color-background)] rounded-md border border-[var(--color-border)]">
                <select value={sense} onChange={e => setSense(e.target.value)} className="w-full p-2 input-field">
                    <option value="">-- Select Sense --</option>
                    {Object.entries(SENSE_CLASSES).map(([name, id]) => <option key={id} value={id}>{name}</option>)}
                </select>
                <select value={age} onChange={e => setAge(e.target.value)} className="w-full p-2 input-field">
                    <option value="">-- Select Age --</option>
                    {Object.entries(AGE_CLASSES).map(([name, id]) => <option key={id} value={id}>{name}</option>)}
                </select>
                <input id="situational-audio-file-input" type="file" onChange={e => setFile(e.target.files[0])} accept="audio/wav, audio/mpeg, audio/mp4" className="w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200" />
                <button type="button" onClick={handleAddClick} className="px-4 py-2 font-semibold text-white bg-blue-500 rounded-lg hover:bg-blue-600 h-fit">+ Add Audio</button>
            </div>
        );
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null); // Reset error on new submission

        if (!bookDetails.file || !bookDetails.Title) {
            setError('Book Title and a .txt file are required.');
            return;
        }

        // Validation for abstracts: if an abstract has a name, it must have a file.
        const abstractsWithNames = aiConfig.target_abstracts.filter(abs => abs.name.trim() !== '');
        if (abstractsWithNames.some(abs => !abs.file)) {
            setError('Each created abstract (with a name) must have an associated audio file uploaded.');
            return;
        }

        setIsLoading(true);

        try {
            const formData = new FormData();

            // 1. Append all book metadata fields
            formData.append('Title', bookDetails.Title);
            formData.append('Synopsis', bookDetails.Synopsis);
            formData.append('Access', bookDetails.Access);
            formData.append('PublishedDate', bookDetails.PublishedDate);
            formData.append('CategoryId', bookDetails.CategoryId);
            formData.append('AuthorId', user.id);
            if (bookDetails.Image) formData.append('Image', bookDetails.Image);
            formData.append('file', bookDetails.file);

            // 2. Append situational audio files
            situationalAudio.forEach(item => {
                formData.append(`situational_audio_${item.senseId}_${item.ageId}`, item.file);
            });

            // 3. Prepare target abstracts, append their audio, and build the dictionary for the JSON
            const targetAbstractsDict = {};
            aiConfig.target_abstracts.forEach(item => {
                if (item.name && item.description && item.file) {
                    targetAbstractsDict[item.name] = item.description;
                    // The key for the audio file must match what the backend expects.
                    // Using the abstract name is a common pattern.
                    formData.append(`abstract_audio_${item.name}`, item.file);
                }
            });

            // 4. Prepare and append the final AI config JSON
            const finalAiConfig = {
                title: bookDetails.Title,
                classifier_driver: 'nn',
                extractor_driver: 'llm',
                llm_ollama_model: 'phi4-mini',
                confidence_threshold: aiConfig.confidence_threshold,
                allowed_senses: aiConfig.allowed_senses,
                allowed_ages: aiConfig.allowed_ages,
                target_abstracts: targetAbstractsDict,
            };
            formData.append('config_json', JSON.stringify(finalAiConfig));

            const job = await authorApi.processNewBook(formData);
            setSelectedJobId(job.jobId);
            setPage('jobDetails');

        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to start processing job.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-4 md:p-6">
            <h1 className="text-3xl font-bold mb-6 text-[var(--color-text-primary)]">Create & Process a New Book</h1>
            <form onSubmit={handleSubmit} className="space-y-8">

                {/* Section 1: Book Metadata */}
                <div className="p-6 bg-[var(--color-background-secondary)] rounded-lg shadow">
                    <h2 className="text-xl font-semibold mb-4 border-b pb-2">1. Book Metadata</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label htmlFor="Title" className="block text-sm font-medium text-[var(--color-text-primary)] mb-1">Book Title</label>
                            <input type="text" name="Title" id="Title" value={bookDetails.Title} onChange={handleDetailChange} required className="w-full p-2 input-field" />
                        </div>
                        <div>
                            <label htmlFor="CategoryId" className="block text-sm font-medium mb-1">Category</label>
                            {isCategoriesLoading ? <p>Loading...</p> : (
                                <select name="CategoryId" id="CategoryId" value={bookDetails.CategoryId} onChange={handleDetailChange} className="w-full p-2 input-field">
                                    {categories.map(cat => <option key={cat.id} value={cat.id}>{cat.categoryName}</option>)}
                                </select>
                            )}
                        </div>
                    </div>
                    <div className="mt-4">
                        <label htmlFor="Synopsis" className="block text-sm font-medium text-[var(--color-text-primary)] mb-1">Synopsis</label>
                        <textarea name="Synopsis" id="Synopsis" value={bookDetails.Synopsis} onChange={handleDetailChange} rows="4" className="w-full p-2 input-field resize-y"></textarea>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                        <div>
                            <label htmlFor="Access" className="block text-sm font-medium mb-1">Access Type</label>
                            <select name="Access" id="Access" value={bookDetails.Access} onChange={handleDetailChange} className="w-full p-2 input-field">
                                <option value="Subscription">Subscription</option>
                                <option value="Free">Free</option>
                            </select>
                        </div>
                        <div>
                            <label htmlFor="PublishedDate" className="block text-sm font-medium mb-1">Published Date</label>
                            <input type="date" name="PublishedDate" id="PublishedDate" value={bookDetails.PublishedDate} onChange={handleDetailChange} required className="w-full p-2 input-field" />
                        </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6 pt-4 border-t border-[var(--color-border)]">
                        <div>
                            <label htmlFor="Image" className="block text-sm font-medium mb-1">Cover Image</label>
                            <input type="file" name="Image" id="Image" onChange={handleFileChange} accept="image/png, image/jpeg" className="w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200" />
                            {coverPreview && <img src={coverPreview} alt="Cover preview" className="mt-4 rounded-md w-32 object-cover aspect-[2/3]" />}
                        </div>
                        <div>
                            <label htmlFor="file" className="block text-sm font-medium mb-1">Book Content (.txt)</label>
                            <input type="file" name="file" id="file" onChange={handleFileChange} accept=".txt" required className="w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200" />
                        </div>
                    </div>
                </div>

                {/* Section 2: AI Config */}
                <div className="p-6 bg-[var(--color-background-secondary)] rounded-lg shadow space-y-4">
                    <h2 className="text-xl font-semibold mb-4 border-b pb-2">2. AI Pipeline Configuration</h2>
                    <div>
                        <label htmlFor="confidence" className="block text-sm font-medium mb-1">Confidence Threshold ({aiConfig.confidence_threshold.toFixed(2)})</label>
                        <input type="range" id="confidence" min="0.5" max="1.0" step="0.01" value={aiConfig.confidence_threshold} onChange={e => handleAiConfigChange('confidence_threshold', parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700" />
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <MultiSelectDropdown label="Allowed Senses" options={SENSE_CLASSES} selectedIds={aiConfig.allowed_senses} onChange={(ids) => handleAiConfigChange('allowed_senses', ids)} />
                        <MultiSelectDropdown label="Allowed Ages" options={AGE_CLASSES} selectedIds={aiConfig.allowed_ages} onChange={(ids) => handleAiConfigChange('allowed_ages', ids)} />
                    </div>
                </div>

                {/* Section 3: Situational Audio */}
                <div className="p-6 bg-[var(--color-background-secondary)] rounded-lg shadow">
                    <h2 className="text-xl font-semibold mb-4 border-b pb-2">3. Situational Audio (Optional)</h2>
                    <p className="text-sm text-gray-500 mb-4">Upload custom audio files for specific Sense/Age combinations. Maximum of 33 unique combinations.</p>
                    <SituationalAudioInput />
                    <div className="mt-4 space-y-2">
                        {situationalAudio.map((item, index) => (
                            <div key={index} className="flex justify-between items-center p-2 bg-[var(--color-background)] rounded text-sm">
                                <span><strong>Sense:</strong> {item.senseName}, <strong>Age:</strong> {item.ageName}</span>
                                <span className="text-gray-500 truncate ml-4">{item.file.name}</span>
                                <button type="button" onClick={() => removeSituationalAudio(index)} className="text-xs text-red-500 hover:text-red-700 ml-4">&times; Remove</button>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Section 4: Target Abstracts */}
                <div className="p-6 bg-[var(--color-background-secondary)] rounded-lg shadow">
                    <h2 className="text-xl font-semibold mb-4 border-b pb-2">4. Target Abstracts</h2>
                    <p className="text-sm text-gray-500 mb-4">Define custom concepts to find in your text. An audio file is <strong>required</strong> for each abstract you create.</p>
                    <div className="space-y-4">
                        {aiConfig.target_abstracts.map((abstract, index) => (
                            <div key={index} className="p-3 bg-[var(--color-background)] rounded-md border border-[var(--color-border)]">
                                <div className="flex justify-between items-center mb-2">
                                    <label className="text-sm font-medium">Abstract #{index + 1}</label>
                                    {aiConfig.target_abstracts.length > 1 && <button type="button" onClick={() => removeAbstract(index)} className="text-xs text-red-500 hover:text-red-700">&times; Remove</button>}
                                </div>
                                <input type="text" value={abstract.name} onChange={e => handleAbstractChange(index, 'name', e.target.value)} placeholder="Abstract Name (e.g., 'dragon')" className="w-full p-2 mb-2 input-field" />
                                <textarea value={abstract.description} onChange={e => handleAbstractChange(index, 'description', e.target.value)} placeholder="Explanation or Description..." className="w-full p-2 input-field min-h-[60px] resize-y" />
                                <div className="mt-2">
                                    <label className="block text-xs font-medium text-[var(--color-text-muted)] mb-1">Associated Audio File (Required)</label>
                                    <input type="file" onChange={e => handleAbstractFileChange(index, e.target.files[0])} accept="audio/wav, audio/mpeg, audio/mp4" required={abstract.name.trim() !== ''} className="w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gray-100 file:text-gray-700 hover:file:bg-gray-200" />
                                </div>
                            </div>
                        ))}
                    </div>
                    <button type="button" onClick={addAbstract} className="mt-4 text-sm font-semibold text-[var(--color-accent)] hover:text-[var(--color-accent-hover)]">+ Add another abstract</button>
                </div>

                {/* Submission */}
                <div className="flex justify-end items-center pt-4 border-t border-[var(--color-border)]">
                    {error && <p className="text-sm text-red-500 mr-4">{error}</p>}
                    <button type="submit" disabled={isLoading} className="px-6 py-2.5 font-semibold text-white bg-[var(--color-accent)] rounded-lg hover:bg-[var(--color-accent-hover)] disabled:opacity-50">
                        {isLoading ? 'Submitting...' : 'Create Book & Start Processing'}
                    </button>
                </div>
            </form>
        </div>
    );
};
export default NewBookPage;