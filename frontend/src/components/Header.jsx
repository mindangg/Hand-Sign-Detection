import React, { useState } from 'react'

import { Link } from 'react-router-dom'

import logo from '../assets/logo.png'

import '../styles/Header.css'

export default function Header() {
    const [toggle, setToggle] = useState('home')

    return (
        <header>
            <div className='header-logo'>
                <img src={logo}></img>
            </div>
            <nav className='header-nav'>
                <Link to='/' className={toggle === 'home' ? 'active' : ''} onClick={() => setToggle('home')}>HOME</Link>
                <Link to='/' className={toggle === 'about' ? 'active' : ''} onClick={() => setToggle('about')}>ABOUT US</Link>
                <Link to='/hand-sign' className={toggle === 'hand' ? 'active' : ''} onClick={() => setToggle('hand')}>HAND SIGN</Link>
            </nav>
            <div>
                <i className="fa-solid fa-hand-holding-heart"></i>
            </div>

                <div className="underline"></div>
        </header>
    )
}
