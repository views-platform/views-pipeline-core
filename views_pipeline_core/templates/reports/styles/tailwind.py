def get_css() -> str:
    """
    Generates the CSS styles for the reports using Tailwind CSS.
    """
    return """
        <!-- Tailwind CSS CDN -->
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            primary: '#6750A4',
                            'on-primary': '#FFFFFF',
                            'primary-container': '#EADDFF',
                            secondary: '#625B71',
                            'on-secondary': '#FFFFFF',
                            'secondary-container': '#E8DEF8',
                            tertiary: '#7D5260',
                            'on-tertiary': '#FFFFFF',
                            'tertiary-container': '#FFD8E4',
                            error: '#B3261E',
                            'on-error': '#FFFFFF',
                            'error-container': '#F9DEDC',
                            outline: '#79747E',
                            background: '#FFFFFF',
                            'on-background': '#1F1F1F',
                            surface: '#FFFFFF',
                            'on-surface': '#1F1F1F',
                            'surface-variant': '#F3EDF7',
                            'on-surface-variant': '#49454F',
                        },
                        fontFamily: {
                            sans: ['Roboto', 'system-ui', 'sans-serif'],
                        },
                        borderRadius: {
                            'sm': '8px',
                            'md': '12px',
                            'lg': '16px',
                            'xl': '28px',
                        },
                        boxShadow: {
                            card: '0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1)',
                            'card-hover': '0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05)',
                        },
                        animation: {
                            'fade-in': 'fadeIn 0.3s ease-in-out',
                        },
                        keyframes: {
                            fadeIn: {
                                '0%': { opacity: '0', transform: 'translateY(10px)' },
                                '100%': { opacity: '1', transform: 'translateY(0)' },
                            }
                        }
                    }
                }
            }
        </script>
        <style>
            /* Custom styles */
            body {
                background-color: #F9FAFB;
                color: #1F1F1F;
            }
            
            .gradient-bar {
                height: 4px;
                background: linear-gradient(90deg, 
                    var(--primary, #6750A4), 
                    var(--secondary, #625B71), 
                    var(--tertiary, #7D5260));
            }
            
            .table-container {
                overflow-x: auto;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
                background: white;
            }
            
            .table-container table {
                min-width: 100%;
            }
            
            .table-container th {
                background-color: #F9FAFB;
                color: #4B5563;
                font-weight: 600;
                text-align: left;
                padding: 0.75rem 1.5rem;
                border-bottom: 1px solid #E5E7EB;
            }
            
            .table-container td {
                padding: 1rem 1.5rem;
                border-bottom: 1px solid #E5E7EB;
                color: #4B5563;
            }
            
            .table-container tr:last-child td {
                border-bottom: none;
            }
            
            /* Add alternating row colors */
            .table-container tbody tr:nth-child(even) {
                background-color: #F9FAFB;
            }
            
            .table-container tbody tr:nth-child(odd) {
                background-color: #FFFFFF;
            }
            
            .table-container tbody tr:hover {
                background-color: #F3F4F6;
            }
            
            .table-header {
                font-size: 1.125rem;
                font-weight: 600;
                color: #1F2937;
                padding: 1rem 1.5rem;
                border-bottom: 1px solid #E5E7EB;
            }
            
            .visualization-card {
                animation: fadeIn 0.3s ease-in-out;
            }
            
            .image-card img {
                transition: transform 0.3s ease;
                max-width: 100%;
                height: auto;
            }
            
            .image-card:hover img {
                transform: scale(1.02);
            }
            
            /* Hyperlink styles */
            a {
                color: var(--primary);
                text-decoration: none;
                font-weight: 500;
                transition: color 0.2s ease;
            }
            
            a:hover {
                color: var(--secondary);
                text-decoration: underline;
            }
            
            /* Ensure proper wrapping */
            .visualization-card > div:not(.gradient-bar),
            .image-card > div:not(.gradient-bar),
            .table-container {
                overflow: hidden;
                max-width: 100%;
            }
            
            /* Split table styles */
            .split-table-container {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1.5rem;
            }
            
            @media (max-width: 768px) {
                .split-table-container {
                    grid-template-columns: 1fr;
                }
            }
            /* Markdown container */
        .markdown-container {
            background-color: #F9FAFB;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        /* Markdown specific styles */
        .markdown-container h1, 
        .markdown-container h2, 
        .markdown-container h3, 
        .markdown-container h4, 
        .markdown-container h5, 
        .markdown-container h6 {
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .markdown-container h1 {
            font-size: 1.875rem; /* 30px */
            color: var(--primary);
        }
        
        .markdown-container h2 {
            font-size: 1.5rem; /* 24px */
            color: var(--secondary);
        }
        
        .markdown-container h3 {
            font-size: 1.25rem; /* 20px */
            color: var(--tertiary);
        }
        
        .markdown-container p {
            margin-bottom: 1rem;
            line-height: 1.625;
        }
        
        .markdown-container ul, 
        .markdown-container ol {
            margin-bottom: 1rem;
            padding-left: 1.5rem;
        }
        
        .markdown-container li {
            margin-bottom: 0.5rem;
        }
        
        .markdown-container strong {
            font-weight: 600;
        }
        
        .markdown-container em {
            font-style: italic;
        }
        
        .markdown-container code {
            font-family: monospace;
            background-color: #f3f4f6;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }
        
        .markdown-container pre {
            background-color: #1f2937;
            color: #f3f4f6;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            margin-bottom: 1rem;
        }
        
        .markdown-container pre code {
            background-color: transparent;
            padding: 0;
        }
        
        .markdown-container blockquote {
            border-left: 4px solid var(--primary);
            padding-left: 1rem;
            margin-left: 0;
            color: #4b5563;
            font-style: italic;
            margin-bottom: 1rem;
        }
        
        .markdown-container table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1rem;
        }
        
        .markdown-container table th {
            background-color: #f9fafb;
            text-align: left;
            padding: 0.75rem;
            border: 1px solid #e5e7eb;
            font-weight: 600;
        }
        
        .markdown-container table td {
            padding: 0.75rem;
            border: 1px solid #e5e7eb;
        }
        
        .markdown-container a {
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s ease;
        }
        
        .markdown-container a:hover {
            color: var(--secondary);
            text-decoration: underline;
        }
        /* Column splitting container - vertical layout */
        .split-table-container-columns {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .table-chunk {
            overflow-x: auto;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
            background: white;
        }
        
        /* Row splitting container */
        .split-table-container-rows {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }
        
        @media (max-width: 768px) {
            .split-table-container-rows {
                grid-template-columns: 1fr;
            }
        }
        
        /* Improved table spacing */
        .table-container table,
        .table-chunk table {
            width: 100% !important;
            table-layout: auto;
            border-collapse: separate;
            border-spacing: 0;
        }
        
        .table-container th,
        .table-container td,
        .table-chunk th,
        .table-chunk td {
            padding: 12px 16px !important;
            border-bottom: 1px solid #e9ecef !important;
            word-break: break-word;
            white-space: normal;
            max-width: 350px;  /* Maximum width for cells */
            min-width: 100px;   /* Minimum width for cells */
        }
        
        .table-container th,
        .table-chunk th {
            background-color: #f8f9fa !important;
            font-weight: 600 !important;
            color: #4B5563;
        }
        
        .table-container tr:last-child td,
        .table-chunk tr:last-child td {
            border-bottom: none !important;
        }
        /* Responsive table headers */
        .table-container th {
            min-width: 150px;  /* Minimum width for column headers */
        }
        
        /* Auto column widths based on content */
        .table-auto-layout {
            table-layout: auto !important;
        }
        </style>
    """