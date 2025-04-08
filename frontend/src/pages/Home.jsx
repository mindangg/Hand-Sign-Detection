import React from 'react'

import img from '../assets/image.png'

import '../styles/Home.css'

export default function Home() {
    return (
        <div className='home'>
            <div>
                <img src={img}></img>
            </div>

            <div>
                <h2>Communicate With Deaf And</h2>
                <h2>Hard Of Hearing People</h2>
            </div>

            <div>
            Talk with your hands—literally!
            Throw a peace sign, point to the sky, or flash an “OK”—our smart AI picks it up instantly using just your webcam. No apps, no setup, just pure hand-sign magic. Whether you're playing, learning, or just having fun, it's all in the palm of your hand!
            </div>
        </div>
    )
}
