import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
// import ToolPage from './pages/ToolPage';

const Routing = () => {


  return (
    <Routes>
      <Route path="/" element={<Home />}/>
      {/* <Route path="/skin-disease-predictor" element={<ToolPage/>}/> */}
    </Routes>
  );
}

export default Routing;
