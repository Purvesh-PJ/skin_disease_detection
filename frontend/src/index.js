import React from 'react';
import ReactDOM from 'react-dom/client';
import Routing from './Routing';
import { BrowserRouter } from 'react-router-dom';
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <BrowserRouter>
    <React.StrictMode>
      <Routing />
    </React.StrictMode>
  </BrowserRouter>
);
