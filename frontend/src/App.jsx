import React from 'react'

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Home from './pages/Home'
import Header from './components/Header';
import HandSign from './pages/HandSign';

export default function App() {
    return (
      <Router>
        <Header/>
        <Routes>
          <Route path='/' element={<Home/>}/>
          <Route path='hand-sign' element={<HandSign/>}/>
          <Route/>
        </Routes>
      </Router>
    )
}
