/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html"],
  theme: {
    extend: {
      colors: {
        sidebar: '#0d233a',
        'sidebar-hover': '#153350',
        'sidebar-active': '#1677ff',
        canvas: '#f4f7fb',
        canvasBg: '#f4f7fb',
        bgSurface: '#f4f7fb',
        cardBg: '#ffffff',
        'card-border': '#e2e8f0',
        'brand-blue': '#1677ff',
        brandBlue: '#0066cc',
        'brand-green': '#12b886',
        'brand-red': '#fa5252',
        'brand-purple': '#7950f2',
        'brand-teal': '#15aabf',
        badgeGreen: '#e6f8ee',
        textGreen: '#1aa260',
        badgeBlue: '#e9f2fe',
        badgePurple: '#f3e8ff',
        textPurple: '#8a2be2',
        badgeAmber: '#fef3e2',
        textAmber: '#f59e0b',
        chartGreen: '#10b981',
        chartRed: '#ef4444',
        chartBlue: '#2563eb',
        chartTeal: '#06b6d4',
        chartOrange: '#f97316',
        chartGray: '#94a3b8',
        chartPurple: '#8b5cf6',
        brand: {
          navy: '#091a2d',
          sidebar: '#0d233a',
          active: '#1677ff',
          dark: '#081726',
          border: '#e2e8f0',
          hover: '#132842',
          blue: '#0284fe',
          'blue-hover': '#0070de',
          lightBlue: '#e8f3ff',
          textMuted: '#8497ac',
          cardBg: '#ffffff',
          pageBg: '#f4f7fb',
          success: '#10b981',
          successLight: '#dcfce7'
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif']
      }
    }
  },
  plugins: []
};
